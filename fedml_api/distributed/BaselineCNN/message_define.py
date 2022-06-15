class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2
    MSG_TYPE_S2C_FINISH = 3

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 5
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 6
    MSG_TYPE_C2S_INIT_REGISTER = 7

    # client to gateway
    MSG_TYPE_C2G_SEND_MODEL_TO_GATEWAY = 8

    # gateway to client
    MSG_TYPE_G2C_SYNC_MODEL_TO_CLIENT = 9

    # gateway to server
    MSG_TYPE_G2S_SEND_MODEL_TO_SERVER = 10

    # server to gateway
    MSG_TYPE_S2G_SYNC_MODEL_TO_GATEWAY = 11

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_LOSS = "loss"
    MSG_ARG_KEY_COMP_DELAY = "comp_delay"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_MODEL_GRADS = "model_grads"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"
    MSG_ARG_KEY_DOWNLOAD_EPOCH = "download_epoch"

    """for sync from and with server"""
    MSG_ARG_KEY_NUM_SAMPLES_DICT = "num_samples_dict"
    MSG_ARG_KEY_LOSS_DICT = "loss_dict"
    MSG_ARG_KEY_MODEL_GRADS_DICT = "model_grads_dict"
    MSG_ARG_KEY_ROUND_DELAY_DICT = "round_delay_dict"
    MSG_ARG_KEY_CLIENT_AVAILABILITY = "client_available"
    MSG_ARG_KEY_CLIENT_ASSOCIATION = "client_association"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"


