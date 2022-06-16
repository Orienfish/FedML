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
    from FedML.fedml_core.distributed.gateway.gateway_manager import GatewayManager
except ModuleNotFoundError: # except ImportError
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import GatewayManager

from .utils import transform_list_to_tensor
from .utils import transform_tensor_to_list


class BaselineCNNGatewayManager(GatewayManager):
    def __init__(self, args, aggregator, logger, comm=None, rank=0, size=0,
                 backend="MQTT", mqtt_host="127.0.0.1", mqtt_port=1883,
                 is_preprocessed=False, batch_selection=None):
        super().__init__(args, comm, rank, size, backend, mqtt_host, mqtt_port)
        self.args = args
        self.aggregator = aggregator
        self.worker_num = args.client_num_in_total
        self.select_num = args.client_num_per_gateway
        self.gateway_round_num = args.gateway_comm_round
        self.gateway_offset = 1  # The ID of first gateway in MQTT
        self.client_offset = args.gateway_num_in_total + 1  # The ID of first client in MQTT
        self.round_delay_limit = args.round_delay_limit
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.batch_selection = batch_selection
        self.rank = rank

        # For client selection
        self.cs = ClientSelection(self.worker_num, args.selection,
                                  self.gateway_round_num, args.cs_gamma,
                                  args.trial_name)

        # For results records
        self.start_time = time.time()
        self.round_start_time = [0.0 for _ in range(self.worker_num)]
        self.round_delay = [0.0 for _ in range(self.worker_num)]
        self.comp_delay = [0.0 for _ in range(self.worker_num)]
        self.round_delay_log = os.path.join(args.result_dir, 'gw{}_round_delay.txt'.format(rank))
        self.comp_delay_log = os.path.join(args.result_dir, 'gw{}_comp_delay.txt'.format(rank))
        self.acc_log = os.path.join(args.result_dir, 'gw{}_acc.txt'.format(rank))
        self.tb_logger = logger

        # Indicator of which client is connected to the gateway
        # The client-gateway association decision is made at the server,
        # so this vector is updated by the server
        self.conn_clients = np.array([False for _ in range(self.worker_num)],
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

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2G_SYNC_MODEL_TO_GATEWAY,
                                              self.handle_message_receive_model_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_FINISH,
                                              self.handle_message_finish_from_server)

        if self.args.method == 'fedavg':
            self.register_message_receive_handler(MyMessage.MSG_TYPE_C2G_SEND_MODEL_TO_GATEWAY,
                                                  self.handle_message_receive_model_from_client_sync)
            self.sync_thread = threading.Thread(target=self.sync_aggregate_trigger)
            self.sync_thread.start()

        elif self.args.method == 'fedasync':
            self.register_message_receive_handler(MyMessage.MSG_TYPE_C2G_SEND_MODEL_TO_GATEWAY,
                                                  self.handle_message_receive_model_from_client_async)

    def handle_message_receive_model_from_client_sync(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.info("handle_message_receive_model_from_client_sync "
                     "sender_id = {}".format(sender_id))

        # Update flag record
        self.flag_client_model_uploaded[sender_id - self.client_offset] = True
        self.flag_available[sender_id - self.client_offset] = True

        # Record the round delay
        round_delay = time.time() - self.round_start_time[sender_id - self.client_offset]
        self.round_delay[sender_id - self.client_offset] = round_delay

        # Receive the information from clients
        cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        cnn_params = transform_list_to_tensor(cnn_params)
        cnn_grads = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        local_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOSS)
        local_comp_delay = msg_params.get(MyMessage.MSG_ARG_KEY_COMP_DELAY)
        # download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        # Record the comp delay
        self.comp_delay[sender_id - self.client_offset] = local_comp_delay

        # Update information for client selection
        self.cs.update_loss_n_delay(local_loss, round_delay, sender_id - self.client_offset)
        self.cs.update_grads(cnn_grads, local_sample_number, sender_id - self.client_offset)

        logging.info("Receive model index = {} "
                     "Received num = {}".format(sender_id - self.client_offset,
                                                sum(self.flag_client_model_uploaded)))
        logging.info('flag uploaded: {}'.format(self.flag_client_model_uploaded))
        self.aggregator.add_local_trained_result(sender_id - self.client_offset,
                                                 cnn_params, local_sample_number)

        self.log_delay()

    def sync_aggregate_trigger(self):
        # A threaded process that periodically checks whether the sync aggregation
        # can be triggered
        while True:
            time.sleep(10)
            uploaded_num = sum(self.flag_client_model_uploaded)
            not_returned = ~np.array(self.flag_available)
            waiting_time_since_start = (time.time() - np.array(self.round_start_time))[not_returned]
            if np.sum(not_returned) > 0:
                logging.info('not returned: {}'.format(not_returned))
                logging.info('sync waiting time: {}'.format(waiting_time_since_start))
                min_waiting_time = np.min(waiting_time_since_start)
            else:
                min_waiting_time = 0.0

            if uploaded_num >= self.select_num or \
                    (uploaded_num > 0 and min_waiting_time >= self.round_delay_limit):
                # All received or exceed time limit
                logging.info('Sync Aggregation!')
                self.sync_aggregate()

                # Reset uploaded flag
                self.flag_client_model_uploaded = [False for _ in range(self.worker_num)]

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
        self.tb_logger.log_value('gw{}_test_loss'.format(self.rank), test_loss, int(cur_time * 1000))
        self.tb_logger.log_value('gw{}_accuracy'.format(self.rank), accuracy, int(cur_time * 1000))

        # Delay and accuracy logger
        self.log(cur_time, test_loss, accuracy)

        # start the next round
        self.round_idx += 1
        if self.round_idx % self.gateway_round_num == 0:
            self.send_model_to_server(0)

        # Client selection for the next gateway round
        select_ids = self.cs.select(self.select_num, self.flag_available,
                                    self.conn_clients)
        if select_ids.size > 0:
            for idx in select_ids:
                self.send_message_sync_model_to_client(idx + self.client_offset,
                                                       self.round_idx)

    def handle_message_receive_model_from_client_async(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.info("handle_message_receive_model_from_client_async "
                     "sender_id = {}".format(sender_id))

        # Update flag record
        self.flag_client_model_uploaded[sender_id - self.client_offset] = True
        self.flag_available[sender_id - self.client_offset] = True

        # Record the round delay
        round_delay = time.time() - self.round_start_time[sender_id - self.client_offset]
        self.round_delay[sender_id - self.client_offset] = round_delay

        # Receive the information from clients
        cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        cnn_params = transform_list_to_tensor(cnn_params)
        cnn_grads = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        local_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOSS)
        local_comp_delay = msg_params.get(MyMessage.MSG_ARG_KEY_COMP_DELAY)
        download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        # Record the comp delay
        self.comp_delay[sender_id - self.client_offset] = local_comp_delay

        # Update information for client selection
        self.cs.update_loss_n_delay(local_loss, round_delay, sender_id - self.client_offset)
        self.cs.update_grads(cnn_grads, local_sample_number, sender_id - self.client_offset)

        logging.info('flag uploaded: {}'.format(self.flag_client_model_uploaded))
        self.log_delay()

        self.round_idx += 1
        staleness = self.round_idx - download_epoch
        self.aggregator.aggregate_async(cnn_params, local_sample_number, staleness)

        # Reset flag uploaded
        self.flag_client_model_uploaded[sender_id - self.client_offset] = False

        test_loss, accuracy = self.aggregator.test_on_server_for_all_clients(self.round_idx,
                                                                             self.batch_selection)
        cur_time = time.time() - self.start_time
        logging.info('Round {} cur time: {} acc: {}'.format(self.round_idx,
                                                            cur_time,
                                                            accuracy))
        logging.info('#########################################\n')

        # Tensoboard logger
        self.tb_logger.log_value('gw{}_test_loss'.format(self.rank), test_loss, int(cur_time * 1000))
        self.tb_logger.log_value('gw{}_accuracy'.format(self.rank), accuracy, int(cur_time * 1000))

        # Delay and accuracy logger
        self.log(cur_time, test_loss, accuracy)

        if self.round_idx % self.gateway_round_num == 0:
            self.send_model_to_server(0)

        # Client selection for the next gateway round
        select_ids = self.cs.select(self.select_num, self.flag_available,
                                    self.conn_clients)
        if select_ids.size > 0:
            for idx in select_ids:
                self.send_message_sync_model_to_client(idx + self.client_offset,
                                                       self.round_idx)

    def send_message_sync_model_to_client(self, receive_id, client_index):
        receive_id = int(receive_id)
        logging.info("send_message_sync_model_to_client. " 
                     "receive_id = {} client_idx = {}".format(receive_id, client_index))
        self.round_start_time[receive_id - self.client_offset] = time.time()

        # Set available to False to prevent client selection
        self.flag_available[receive_id - self.client_offset] = False
        
        global_model_params = self.aggregator.get_global_model_params()

        global_model_params = transform_tensor_to_list(global_model_params)

        message = Message(MyMessage.MSG_TYPE_G2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH, self.round_idx)
        self.send_message(message)

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info('*****************************************')
        logging.info("handle_message_receive_model_from_server.")
        global_cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        global_cnn_params = transform_list_to_tensor(global_cnn_params)
        self.aggregator.set_global_model_params(global_cnn_params)

        # self.flag_available = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_AVAILABILITY)
        conn_ids = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ASSOCIATION)
        self.conn_clients = np.array([False for _ in range(self.worker_num)],
                                     dtype=np.bool)
        self.conn_clients[conn_ids] = True
        self.download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        round_delay_dict = msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_DELAY_DICT)
        loss_dict = msg_params.get(MyMessage.MSG_ARG_KEY_LOSS_DICT)
        grads_dict = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS_DICT)
        num_samples_dict = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES_DICT)

        logging.info('conn ids: {}'.format(conn_ids))
        logging.info('loss dict: {}'.format(loss_dict))
        logging.info('grads dict {}'.format([grad[:2] for grad in grads_dict.values()]))
        logging.info('round delay dict: {}'.format(round_delay_dict))
        logging.info('*****************************************\n')

        for k in round_delay_dict.keys():
            logging.info('update client {}'.format(k))
            self.cs.update_loss_n_delay(loss_dict[k], round_delay_dict[k], int(k))
            self.cs.update_grads(grads_dict[k], num_samples_dict[k], int(k))

        # Client selection for the next gateway round
        select_ids = self.cs.select(self.select_num, self.flag_available,
                                    self.conn_clients)
        if select_ids.size > 0:
            for idx in select_ids:
                self.send_message_sync_model_to_client(idx + self.client_offset,
                                                       self.round_idx)

    def send_model_to_server(self, receive_id):
        logging.info('*****************************************')
        logging.info('send model to server!')
        receive_id = int(receive_id)
        global_model_params = self.aggregator.get_global_model_params()
        global_model_params = transform_tensor_to_list(global_model_params)

        # Build a dictionary and only report for the devices that
        # are connected to the current gateway and are available
        round_delay_dict, loss_dict, grads_dict, num_samples_dict = {}, {}, {}, {}
        conn_ids = np.arange(self.client_num, dtype=np.int32)[self.conn_clients]
        conn_ids = [int(idx) for idx in list(conn_ids)]

        logging.info('conn ids: {}'.format(conn_ids))

        for idx in conn_ids:
            round_delay_dict[idx] = self.cs.est_delay[idx]
            loss_dict[idx] = self.cs.losses[idx]
            grads_dict[idx] = list(self.cs.grads[idx])
            num_samples_dict[idx] = int(self.cs.num_samples[idx])

        logging.info('loss dict: {}'.format(loss_dict))
        logging.info('grads dict {}'.format([grad[:2] for grad in grads_dict.values()]))
        logging.info('round delay dict: {}'.format(round_delay_dict))
        logging.info('*****************************************\n')

        message = Message(MyMessage.MSG_TYPE_G2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_GRADS_DICT, grads_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES_DICT, num_samples_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_LOSS_DICT, loss_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_DELAY_DICT, round_delay_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH, self.download_epoch)
        self.send_message(message)

    def handle_message_finish_from_server(self, msg_params):
        logging.info("handle_message_finish from server.")
        self.finish()

    def log_delay(self):
        # Log the round and comp delays in the latest round
        with open(self.round_delay_log, 'a+') as out:
            np.savetxt(out, np.array(self.round_delay).reshape((1, -1)),
                       delimiter=',')
        self.round_delay = [0.0 for _ in range(self.worker_num)]

        with open(self.comp_delay_log, 'a+') as out:
            np.savetxt(out, np.array(self.comp_delay).reshape((1, -1)),
                       delimiter=',')
        self.comp_delay = [0.0 for _ in range(self.worker_num)]

    def log(self, cur_time, test_loss, accuracy):
        # Log the test loss and accuracy
        with open(self.acc_log, 'a+') as out:
            out.write('{},{},{},{}\n'.format(self.round_idx, cur_time,
                                             test_loss, accuracy))

