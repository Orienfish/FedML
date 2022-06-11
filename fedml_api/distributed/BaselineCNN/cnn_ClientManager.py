import logging
import time

#sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
#sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor
from .utils import transform_tensor_to_list

from FedML.fedml_core.distributed.communication.message import Message
from FedML.fedml_core.distributed.communication.mpi.com_manager import MpiCommunicationManager
from FedML.fedml_core.distributed.communication.mqtt.mqtt_comm_manager import MqttCommManager

# This is a simple extension to the basic MqttCommManager
# The reason to use this new MqttCommManager is that, when the client disconnects
# for some reason, then after it attempts to connect again, the client will automatically
# send another init register to server
class myMqttCommManager(MqttCommManager):
    def __init__(self, host, port, client_mgr, topic='fedml', client_id=0,
                 client_num=0):
        self.client_mgr = client_mgr
        super().__init__(host, port, topic=topic,
                         client_id=client_id, client_num=client_num)

    def _on_connect(self, client, userdata, flags, rc):
        """
            [server]
            sending message topic (publish): serverID_clientID
            receiving message topic (subscribe): clientID

            [client]
            sending message topic (publish): clientID
            receiving message topic (subscribe): serverID_clientID

        """
        logging.info("_on_connect: Connection returned with result code: {}".format(str(rc)))
        # subscribe one topic
        if self.client_id == 0:
            # server
            for client_ID in range(1, self.client_num + 1):
                result, mid = self._client.subscribe(self._topic + str(client_ID), 0)
                self._unacked_sub.append(mid)
                # print(result)
        else:
            # client
            result, mid = self._client.subscribe(self._topic + str(0) + "_" + str(self.client_id), 0)
            self._unacked_sub.append(mid)
            # print(result)

            # This is the major difference for the client
            self.client_mgr.send_register_to_server()


class BaseCNNClientManager(ClientManager):
    def __init__(self, mqtt_port, mqtt_host, args, trainer, comm=None, rank=0, size=0, backend="MQTT"):
        self.args = args
        self.size = size
        self.rank = rank

        self.backend = backend
        if backend == "MPI":
            self.com_manager = MpiCommunicationManager(comm, rank, size,
                                                       node_type="server")
        elif backend == "MQTT":
            HOST = mqtt_host
            # HOST = "broker.emqx.io"
            PORT = mqtt_port
            self.com_manager = myMqttCommManager(HOST, PORT, self, client_id=rank,
                                                 client_num=size - 1)
        else:
            self.com_manager = MpiCommunicationManager(comm, rank, size,
                                                       node_type="client")
        self.com_manager.add_observer(self)
        self.message_handler_dict = dict()

        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_FINISH,
                                              self.handle_message_finish_from_server)

    def handle_message_init_from_server(self, msg_params):
        logging.info("handle_message_init_from_server.")
        global_cnn_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.download_epoch = msg_params.get(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH)

        global_cnn_params = transform_list_to_tensor(global_cnn_params)

        self.trainer.update_model(global_cnn_params)
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

    def handle_message_finish_from_server(self, msg_params):
        logging.info("handle_message_finish from server.")
        self.finish()

    def send_register_to_server(self):
        logging.info("init_register_to_server.")
        message = Message(MyMessage.MSG_TYPE_C2S_INIT_REGISTER,
                          self.get_sender_id(), 0)
        self.send_message(message)

    def send_model_to_server(self, receive_id, cnn_params, cnn_grads, local_sample_num,
                             local_loss, local_comp_delay):
        
        cnn_params = transform_tensor_to_list(cnn_params)

        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, cnn_params)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_GRADS, cnn_grads)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_LOSS, local_loss)
        message.add_params(MyMessage.MSG_ARG_KEY_COMP_DELAY, local_comp_delay)
        message.add_params(MyMessage.MSG_ARG_KEY_DOWNLOAD_EPOCH, self.download_epoch)
        self.send_message(message)


    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        start = time.time()
        cnn_params, cnn_grads, local_sample_num, local_loss = self.trainer.train()
        local_comp_delay = time.time() - start
        self.send_model_to_server(0, cnn_params, cnn_grads, local_sample_num,
                                  local_loss, local_comp_delay)
