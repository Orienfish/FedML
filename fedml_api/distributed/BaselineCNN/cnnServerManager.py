import logging
import os
import sys
import torch
import time
import numpy as np
import signal
import threading

from .message_define import MyMessage
from .clientAssociation import ClientAssociation

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
        time.sleep(20)
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
        self.select_num = args.client_num_per_gateway
        self.gateway_num = args.gateway_num_in_total
        self.round_num = args.comm_round
        self.round_delay_limit = args.round_delay_limit  # Only used in warmup
        self.adjust_round = args.adjust_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.batch_selection = batch_selection

        # For client selection
        self.ca = ClientAssociation(self.worker_num, self.gateway_num, args.association,
                                    args.ca_phi, args.trial_name)

        # For results records
        self.start_time = time.time()
        self.round_start_time = [0.0 for _ in range(self.worker_num)]  # Used in warmup
        #self.round_delay = [0.0 for _ in range(self.worker_num)]
        #self.comp_delay = [0.0 for _ in range(self.worker_num)]
        #self.round_delay_log = os.path.join(args.result_dir, 'round_delay.txt')
        #self.comp_delay_log = os.path.join(args.result_dir, 'comp_delay.txt')
        self.acc_log = os.path.join(args.result_dir, 'acc.txt')
        self.tb_logger = logger

        # Indicator of which client is connected to the gateway
        # The client-gateway association decision is made at the server,
        # so this vector is updated by the server
        self.conn_clients = np.zeros((self.worker_num, self.gateway_num),
                                     dtype=np.bool)

        # Indicator of the status of the clients
        # Since the gateway can only know the status of the clients that is connected to it,
        # this vector is periodically synced with the server
        self.flag_available = np.array([True for _ in range(self.worker_num)],
                                       dtype=np.bool)

        # Indicator of which client has uploaded in sync aggregation
        # This is related but different from the availability of the clients
        # If True, client has uploaded the model and finished last round, thus available
        # If False, the local round has not returned, thus unavailable
        # Formally, flag_available >= flag_client_model_uploaded
        self.flag_client_model_uploaded = np.array([False for _ in range(self.worker_num)],
                                                   dtype=np.bool)

        # Start from warmup
        self.warmup_done = False

    def run(self):
        super().run()
        # Create a thread to monitor running status and kill the program at the end
        x = threading.Thread(target=terminate)
        x.start()

    def send_init_msg(self):
        global_model_params = self.aggregator.get_global_model_params()
        for receiver_id in range(1, self.client_num):
            self.send_message_init_to_client(receiver_id, global_model_params, 0)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_INIT_REGISTER,
                                              self.handle_init_register_from_client)
        self.warmup_thread = threading.Thread(target=self.warmup_checker)
        self.warmup_thread.start()

        if self.args.method == 'fedavg':
            self.register_message_receive_handler(MyMessage.MSG_TYPE_G2S_SEND_MODEL_TO_SERVER,
                                                  self.handle_message_receive_model_from_gateway_sync)

        elif self.args.method == 'fedasync':
            self.register_message_receive_handler(MyMessage.MSG_TYPE_G2S_SEND_MODEL_TO_SERVER,
                                                  self.handle_message_receive_model_from_gateway_async)

    def handle_message_receive_model_from_gateway_sync(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.info("handle_message_receive_model_from_client_sync "
                     "sender_id = {}".format(sender_id))

        # Update flag record
        self.flag_client_model_uploaded[sender_id - 1] = True
        # self.flag_available[sender_id - 1] = True

        # Receive the information from clients
        cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        cnn_params = transform_list_to_tensor(cnn_params)
        # cnn_grads = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS)
        # local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        # local_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOSS)
        # local_comp_delay = msg_params.get(MyMessage.MSG_ARG_KEY_COMP_DELAY)
        # download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        round_delay_dict = msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_DELAY_DICT)
        # loss_dict = msg_params.get(MyMessage.MSG_ARG_KEY_LOSS_DICT)
        grads_dict = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS_DICT)
        num_samples_dict = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES_DICT)

        for k in round_delay_dict.keys():
            self.ca.update_delay_n_rate(round_delay_dict[k], k)
            self.ca.update_grads(grads_dict[k], num_samples_dict[k], k)

        logging.info("Receive model index = {} "
                     "Received num = {}".format(sender_id - 1, sum(self.flag_client_model_uploaded)))
        logging.info('flag uploaded: {}'.format(self.flag_client_model_uploaded))
        local_sample_number = sum([num_samples_dict[k] for k in num_samples_dict.keys()])
        self.aggregator.add_local_trained_result(sender_id - 1, cnn_params, local_sample_number)

        if sum(self.flag_client_model_uploaded) >= self.gateway_num:
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

            # Update the device-gatewy association
            if self.round_idx % self.adjust_round == 0:
                self.conn_clients = self.ca.solve()

            if self.round_idx == self.round_num or accuracy >= self.args.target_accuracy:
                for receiver_id in range(1, self.client_num):
                    self.send_message_finish_to_client(receiver_id)
                global running
                running = False
            else:
                # gateway trigger
                for gateway_id in range(self.gateway_num):
                    self.send_message_sync_model_to_gateway(gateway_id,
                                                            self.round_idx)

    def warmup_checker(self):
        # A threaded process that periodically checks whether the warmup is done
        while True:
            time.sleep(20)
            uploaded_num = sum(self.flag_client_model_uploaded)
            not_returned = ~np.array(self.flag_available)
            waiting_time_since_start = (time.time() - np.array(self.round_start_time))[not_returned]
            if np.sum(not_returned) > 0:
                logging.info('not returned: {}'.format(not_returned))
                logging.info('warmup waiting time: {}'.format(waiting_time_since_start))
                min_waiting_time = np.min(waiting_time_since_start)
            else:
                min_waiting_time = 0.0

            if uploaded_num >= self.worker_num or \
                    (uploaded_num > 0 and min_waiting_time >= self.round_delay_limit):
                # All received or exceed time limit
                # Start the first round from client association
                self.conn_clients = self.ca.solve()

                # gateway trigger
                for gateway_id in range(self.gateway_num):
                    self.send_message_sync_model_to_gateway(gateway_id,
                                                            self.round_idx)

                break  # End the thread

        # Reset uploaded flag
        self.flag_client_model_uploaded = [False for _ in range(self.worker_num)]

        self.warmup_done = True
        logging.info('All received. Warmup done.')
        logging.info('Start the experiment!')

    def handle_message_receive_model_from_gateway_async(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.info("handle_message_receive_model_from_client_async "
                     "sender_id = {}".format(sender_id))

        # Update flag record
        self.flag_client_model_uploaded[sender_id - 1] = True
        self.flag_available[sender_id - 1] = True

        # Receive the information from clients
        cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        cnn_params = transform_list_to_tensor(cnn_params)
        # cnn_grads = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS)
        # local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        # local_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOSS)
        # local_comp_delay = msg_params.get(MyMessage.MSG_ARG_KEY_COMP_DELAY)
        download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        round_delay_dict = msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_DELAY_DICT)
        # loss_dict = msg_params.get(MyMessage.MSG_ARG_KEY_LOSS_DICT)
        grads_dict = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS_DICT)
        num_samples_dict = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES_DICT)
        local_sample_number = sum([num_samples_dict[k] for k in num_samples_dict.keys()])

        for k in round_delay_dict.keys():
            self.ca.update_delay_n_rate(round_delay_dict[k], k)
            self.ca.update_grads(grads_dict[k], num_samples_dict[k], k)

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

            # Update the device-gatewy association
            if self.round_idx % self.adjust_round == 0:
                self.conn_clients = self.ca.solve()

            if self.round_idx >= self.round_num or accuracy >= self.args.target_accuracy:
                for receiver_id in range(1, self.client_num):
                    self.send_message_finish_to_client(receiver_id)
                global running
                running = False
            else:
                self.send_message_sync_model_to_gateway(sender_id-100,
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

        # Set available to False to prevent client selection
        self.flag_available[receive_id - 1] = False

        global_model_params = transform_tensor_to_list(global_model_params)

        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH, self.round_idx)
        self.send_message(message)

    def send_message_sync_model_to_gateway(self, gateway_id, client_index):
        receive_id = int(gateway_id)
        logging.info("send_message_sync_model_to_client. " 
                     "receive_id = {} client_idx = {}".format(receive_id, client_index))
        self.round_start_time[receive_id-1] = time.time()

        # Set available to False to prevent client selection
        #self.flag_available[receive_id - 1] = False

        global_model_params = self.aggregator.get_global_model_params()
        global_model_params = transform_tensor_to_list(global_model_params)
        update_ids = self.conn_clients[:, gateway_id]

        round_delay_dict, loss_dict, grads_dict, num_samples_dict = {}, {}, {}, {}
        for idx in update_ids:
            round_delay_dict[idx] = self.ca.est_delay[idx]
            loss_dict[idx] = self.ca.losses[idx]
            grads_dict[idx] = self.ca.grads[idx]
            num_samples_dict[idx] = self.ca.num_samples[idx]

        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_ASSOCIATION, update_ids)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH, self.round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_GRADS_DICT, grads_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES_DICT, num_samples_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_LOSS_DICT, loss_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_DELAY_DICT, round_delay_dict)
        self.send_message(message)

    def send_message_finish_to_client(self, receive_id):
        logging.info("send_message_finish_to_client. "
                     "receive_id = {}".format(receive_id))

        message = Message(MyMessage.MSG_TYPE_S2C_FINISH, self.get_sender_id(), receive_id)
        self.send_message(message)

    def log(self, cur_time, test_loss, accuracy):
        # Log the test loss and accuracy
        with open(self.acc_log, 'a+') as out:
            out.write('{},{},{},{}\n'.format(self.round_idx, cur_time,
                                             test_loss, accuracy))

