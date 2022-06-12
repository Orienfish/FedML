import logging
import os
import sys
import torch
import time
import numpy as np
import signal
import threading

from .message_define import MyMessage
from .clientSelection import ClientSelection

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))
try:
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_core.distributed.server.server_manager import ServerManager
except ModuleNotFoundError: # except ImportError
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import ServerManager

from .utils import transform_list_to_tensor
from .utils import transform_tensor_to_list

running = True
def terminate():
    while True:
        time.sleep(10)
        # print('hello')
        if not running:
            try:
                for line in os.popen('ps aux | grep app_CNN.py | grep -v grep'):
                    fields = line.split()
                    pid = fields[1]
                    print('extracted pid: ', pid)
                    os.kill(int(pid), signal.SIGKILL)
            except:
                print('Error encountered while running killing script')


class BaselineCNNServerManager(ServerManager):
    def __init__(self, args, aggregator, logger, comm=None, rank=0, size=0,
                 backend="MQTT", mqtt_host="127.0.0.1", mqtt_port=1883,
                 is_preprocessed=False, batch_selection=None):
        super().__init__(args, comm, rank, size, backend, mqtt_host, mqtt_port)
        self.args = args
        self.aggregator = aggregator
        self.worker_num = args.client_num_in_total
        self.select_num = args.client_num_per_round
        self.round_num = args.comm_round
        self.round_delay_limit = args.round_delay_limit
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.batch_selection = batch_selection

        # For client selection
        self.cs = ClientSelection(self.worker_num, args.selection, self.round_num, args.cs_gamma)

        # For results records
        self.start_time = time.time()
        self.round_start_time = [0.0 for _ in range(self.worker_num)]
        self.round_delay = [0.0 for _ in range(self.worker_num)]
        self.comp_delay = [0.0 for _ in range(self.worker_num)]
        self.round_delay_log = os.path.join(args.result_dir, 'round_delay.txt')
        self.comp_delay_log = os.path.join(args.result_dir, 'comp_delay.txt')
        self.acc_log = os.path.join(args.result_dir, 'acc.txt')
        self.tb_logger = logger

        self.flag_available = [False for _ in range(self.worker_num)]
        # Indicator of which client has uploaded in sync aggregation
        # This is related but different from the availability of the clients
        # If True, client has uploaded the model and finished last round, thus available
        # If False, the local round has not returned, thus unavailable
        # Formally, flag_available >= flag_client_model_uploaded
        self.flag_client_model_uploaded = [False for _ in range(self.worker_num)]

        # Start from warmup
        self.warmup_done = False

    def run(self):
        super().run()
        # Create a thread to monitor running status and kill the program at the end
        x = threading.Thread(target=terminate)
        x.start()

    def send_init_msg(self):
        global_model_params = self.aggregator.get_global_model_params()
        for receiver_id in range(1, self.size):
            self.send_message_init_to_client(receiver_id, global_model_params, 0)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_INIT_REGISTER,
                                              self.handle_init_register_from_client)
        self.warmup_thread = threading.Thread(target=self.warmup_checker)
        self.warmup_thread.start()

        if self.args.method == 'fedavg':
            self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                                  self.handle_message_receive_model_from_client_sync)

            # Create a thread to trigger sync aggregation
            self.sync_agg = threading.Thread(target=self.sync_aggregate_trigger)
            self.sync_agg.start()

        elif self.args.method == 'fedasync':
            self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                                  self.handle_message_receive_model_from_client_async)

    def handle_message_receive_model_from_client_sync(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.info("handle_message_receive_model_from_client_sync "
                     "sender_id = {}".format(sender_id))

        # Update flag record
        self.flag_client_model_uploaded[sender_id - 1] = True
        self.flag_available = True

        # Record the round delay
        round_delay = time.time() - self.round_start_time[sender_id - 1]
        self.round_delay[sender_id - 1] = round_delay

        # Receive the information from clients
        cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        cnn_params = transform_list_to_tensor(cnn_params)
        cnn_grads = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        local_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOSS)
        local_comp_delay = msg_params.get(MyMessage.MSG_ARG_KEY_COMP_DELAY)
        # download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        # Record the comp delay
        self.comp_delay[sender_id - 1] = local_comp_delay

        # Update information for client selection
        self.cs.update_loss_n_delay(local_loss, round_delay, sender_id - 1)
        self.cs.update_grads(cnn_grads, local_sample_number, sender_id - 1)

        logging.info("Receive model index = {} "
                     "Received num = {}".format(sender_id - 1, sum(self.flag_client_model_uploaded)))
        logging.info('flag uploaded: {}'.format(self.flag_client_model_uploaded))
        self.aggregator.add_local_trained_result(sender_id - 1, cnn_params, local_sample_number)


    def sync_aggregate_trigger(self):
        # A threaded process that periodically checks whether the sync aggregation
        # can be triggered
        while True:
            time.sleep(10)
            received_num = sum(self.flag_client_model_uploaded)
            lastest_round_start_time = max(self.round_start_time)

            if self.warmup_done and (received_num >= self.select_num or
                                     (received_num > 0 and
                                      time.time() - lastest_round_start_time >= self.round_delay_limit)):
                logging.info('Sync Aggregation!')
                self.sync_aggregate()

                # Reset uploaded flag
                self.flag_client_model_uploaded = [False for _ in range(self.worker_num)]

    def warmup_checker(self):
        # A threaded process that periodically checks whether the warmup is done
        while True:
            time.sleep(10)
            received_num = sum(self.flag_client_model_uploaded)

            if received_num >= self.worker_num:  # all received
                # Start the first round from client selection
                select_ids = self.cs.select(self.select_num, self.flag_available)
                if select_ids.size > 0:
                    for idx in select_ids:
                        self.send_message_sync_model_to_client(idx + 1,
                                                               self.round_idx)

                break  # End the thread

        # Reset uploaded flag
        self.flag_client_model_uploaded = [False for _ in range(self.worker_num)]

        self.warmup_done = True
        logging.info('All received. Warmup done.')
        logging.info('Start the experiment!')

    def sync_aggregate(self):
        # Sync aggregation
        global_model_params = self.aggregator.aggregate(self.flag_client_model_uploaded)

        # Test
        test_loss, accuracy = self.aggregator.test_on_server_for_all_clients(self.round_idx,
                                                                  self.batch_selection)
        cur_time = time.time() - self.start_time
        logging.info('Round {} cur time: {} acc: {}'.format(self.round_idx,
                                                            cur_time,
                                                            accuracy))
        logging.info('#########################################\n')

        # Tensoboard logger
        self.tb_logger.log_value('test_loss', test_loss, int(cur_time * 1000))
        self.tb_logger.log_value('accuracy', accuracy, int(cur_time * 1000))

        # Delay and accuracy logger
        self.log(cur_time, test_loss, accuracy)

        # start the next round
        self.round_idx += 1
        if self.round_idx == self.round_num or accuracy >= self.args.target_accuracy:
            for receiver_id in range(1, self.size):
                self.send_message_finish_to_client(receiver_id)
            global running
            running = False
        else:
            # Client selection
            select_ids = self.cs.select(self.select_num, self.flag_available)
            if select_ids.size > 0:
                for idx in select_ids:
                    self.send_message_sync_model_to_client(idx + 1,
                                                           self.round_idx)

    def handle_message_receive_model_from_client_async(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.info("handle_message_receive_model_from_client_async "
                     "sender_id = {}".format(sender_id))

        # Update flag record
        self.flag_client_model_uploaded[sender_id - 1] = True
        self.flag_available = True

        # Record the round delay
        round_delay = time.time() - self.round_start_time[sender_id - 1]
        self.round_delay[sender_id - 1] = round_delay

        # Receive the information from clients
        cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        cnn_params = transform_list_to_tensor(cnn_params)
        cnn_grads = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        local_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOSS)
        local_comp_delay = msg_params.get(MyMessage.MSG_ARG_KEY_COMP_DELAY)
        download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        # Record the comp delay
        self.comp_delay[sender_id - 1] = local_comp_delay

        # Update information for client selection
        self.cs.update_loss_n_delay(local_loss, round_delay, sender_id - 1)
        self.cs.update_grads(cnn_grads, local_sample_number, sender_id - 1)

        logging.info('flag uploaded: {}'.format(self.flag_client_model_uploaded))

        if self.warmup_done:
            self.round_idx += 1
            staleness = self.round_idx - download_epoch
            self.aggregator.aggregate_async(cnn_params, local_sample_number, staleness)

            # Reset flag uploaded
            self.flag_client_model_uploaded[sender_id - 1] = False

            test_loss, accuracy = self.aggregator.test_on_server_for_all_clients(self.round_idx,
                                                                                 self.batch_selection)
            cur_time = time.time() - self.start_time
            logging.info('Round {} cur time: {} acc: {}'.format(self.round_idx,
                                                                cur_time,
                                                                accuracy))
            logging.info('#########################################\n')

            # Tensoboard logger
            self.tb_logger.log_value('test_loss', test_loss, int(cur_time * 1000))
            self.tb_logger.log_value('accuracy', accuracy, int(cur_time * 1000))

            # Delay and accuracy logger
            self.log(cur_time, test_loss, accuracy)

            if self.round_idx >= self.round_num or accuracy >= self.args.target_accuracy:
                for receiver_id in range(1, self.size):
                    self.send_message_finish_to_client(receiver_id)
                global running
                running = False
            else:
                # Client selection
                select_ids = self.cs.select(1, self.flag_available)
                if select_ids.size > 0:
                    for idx in select_ids:
                        self.send_message_sync_model_to_client(idx + 1,
                                                               self.round_idx)

    def handle_init_register_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.info("handle_init_register_from_client "
                     "sender_id = {}".format(sender_id))

        # Sync the same initial global model with the client
        global_model_params = self.aggregator.get_global_model_params()
        self.send_message_init_to_client(sender_id, global_model_params, 0)

    def send_message_init_to_client(self, receive_id, global_model_params, client_index):
        logging.info("send_message_init_to_client. "
                     "receive_id = {} client_idx = {}".format(receive_id, client_index))
        self.round_start_time[receive_id - 1] = time.time()
        global_model_params = transform_tensor_to_list(global_model_params)

        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH, self.round_idx)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, client_index):
        receive_id = int(receive_id)
        logging.info("send_message_sync_model_to_client. " 
                     "receive_id = {} client_idx = {}".format(receive_id, client_index))
        self.round_start_time[receive_id-1] = time.time()

        # Set available to False to prevent client selection
        self.flag_available[receive_id - 1] = False
        
        global_model_params = self.aggregator.get_global_model_params()

        global_model_params = transform_tensor_to_list(global_model_params)

        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH, self.round_idx)
        self.send_message(message)

    def send_message_finish_to_client(self, receive_id):
        logging.info("send_message_finish_to_client. "
                     "receive_id = {}".format(receive_id))

        message = Message(MyMessage.MSG_TYPE_S2C_FINISH, self.get_sender_id(), receive_id)
        self.send_message(message)

    def log(self, cur_time, test_loss, accuracy):
        # Log the round and comp delays in the latest round
        with open(self.round_delay_log, 'a+') as out:
            np.savetxt(out, np.array(self.round_delay).reshape((1, -1)),
                       delimiter=',')
        self.round_delay = [0.0 for _ in range(self.worker_num)]

        with open(self.comp_delay_log, 'a+') as out:
            np.savetxt(out, np.array(self.comp_delay).reshape((1, -1)),
                       delimiter=',')
        self.comp_delay = [0.0 for _ in range(self.worker_num)]

        # Log the test loss and accuracy
        with open(self.acc_log, 'a+') as out:
            out.write('{},{},{},{}\n'.format(self.round_idx, cur_time,
                                             test_loss, accuracy))

