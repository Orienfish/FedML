import logging
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor
from .utils import transform_tensor_to_list


class BaseCNNClientManager(ClientManager):
    def __init__(self, mqtt_port, mqtt_host, args, trainer, comm=None, rank=0, size=0, backend="MQTT"):
        super().__init__(args, comm, rank, size, backend, mqtt_host, mqtt_port)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)


    def handle_message_init(self, msg_params):
        logging.info("handle_message_init_from_server.")
        global_cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        global_cnn_params = transform_list_to_tensor(global_cnn_params)

        self.trainer.update_model(global_cnn_params)

    
    def start_training(self):
        self.round_idx = 0
        self.__train()
    

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        global_cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)
        
        global_cnn_params = transform_list_to_tensor(global_cnn_params)

        self.trainer.update_model(global_cnn_params)

        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            self.finish()

    def init_register_to_server(self):
        logging.info("init_register_to_server.")
        message = Message(MyMessage.MSG_TYPE_C2S_INIT_REGISTER, self.get_sender_id(), 0)
        self.send_message(message)

    def send_model_to_server(self, receive_id, cnn_params, local_sample_num):
        
        cnn_params = transform_tensor_to_list(cnn_params)

        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, cnn_params)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH, self.download_epoch)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        cnn_params, local_sample_num = self.trainer.train()
        self.send_model_to_server(0, cnn_params, local_sample_num)
