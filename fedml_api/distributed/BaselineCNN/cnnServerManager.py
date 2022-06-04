import logging
import os
import sys
import torch
import time
import numpy as np
import signal
import threading

from .message_define import MyMessage

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
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.batch_selection = batch_selection

        # For results records
        self.start_time = time.time()
        self.round_start_time = [0.0 for _ in range(self.worker_num)]
        self.round_delay = [0.0 for _ in range(self.worker_num)]
        self.delay_log = os.path.join(args.result_dir, 'delay.txt')
        self.acc_log = os.path.join(args.result_dir, 'acc.txt')
        self.tb_logger = logger

        self.finish = [False for _ in range(self.worker_num)]

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

        if self.args.method == 'fedavg':
            self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                                  self.handle_message_receive_model_from_client_sync)
        elif self.args.method == 'fedasync':
            self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                                  self.handle_message_receive_model_from_client_async)

    def handle_message_receive_model_from_client_sync(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.info("handle_message_receive_model_from_client_sync "
                     "sender_id = {}".format(sender_id))

        # Record the round delay
        self.round_delay[sender_id - 1] = time.time() - self.round_start_time[sender_id - 1]

        # Receive the information from clients
        cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        cnn_params = transform_list_to_tensor(cnn_params)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        # download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        self.aggregator.add_local_trained_result(sender_id - 1, cnn_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        # logging.info("b_all_received = " + str(b_all_received))

        if b_all_received:
            global_model_params = self.aggregator.aggregate()
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
            if self.round_idx == self.round_num:
                for receiver_id in range(1, self.size):
                    self.send_message_finish_to_client(receiver_id)
                global running
                running = False
            else:
                client_indexes = [self.round_idx] * self.args.client_num_per_round

                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id,
                                                           client_indexes[receiver_id - 1])

    def handle_message_receive_model_from_client_async(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.info("handle_message_receive_model_from_client_async "
                     "sender_id = {}".format(sender_id))

        # Record the round delay
        self.round_delay[sender_id - 1] = time.time() - self.round_start_time[sender_id - 1]

        # Receive the information from clients
        cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        cnn_params = transform_list_to_tensor(cnn_params)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        self.round_idx += 1
        staleness = self.round_idx - download_epoch
        self.aggregator.aggregate_async(cnn_params, local_sample_number, staleness)

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

        if self.round_idx >= self.round_num:
            for receiver_id in range(1, self.size):
                self.send_message_finish_to_client(receiver_id)
            global running
            running = False
        else:
            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(receiver_id,
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
        logging.info("send_message_sync_model_to_client. " 
                     "receive_id = {} client_idx = {}".format(receive_id, client_index))
        self.round_start_time[receive_id-1] = time.time()
        
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
        # Log the round delays in the latest round
        with open(self.delay_log, 'a+') as out:
            np.savetxt(out, np.array(self.round_delay).reshape((1, -1)),
                       delimiter=',')
        self.round_delay = [0.0 for _ in range(self.args.client_num_in_total)]

        # Log the test loss and accuracy
        with open(self.acc_log, 'a+') as out:
            out.write('{},{},{},{}\n'.format(self.round_idx, cur_time,
                                             test_loss, accuracy))

