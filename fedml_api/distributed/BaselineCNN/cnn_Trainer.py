from .utils import transform_tensor_to_list


class BaseCNN_Trainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

        self.device = device
        self.args = args

    def update_model(self, cnn_params):
        self.trainer.set_model_params(cnn_params)


    """ not used
    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]
    """

    def train(self):
        old_cnn_params = self.trainer.get_model_params()
        self.trainer.train(self.train_local, self.args)

        cnn_params = self.trainer.get_model_params()
        #cnn_grads = self.trainer.get_model_grads()
        cnn_grads = self.trainer.get_delta_weights(old_cnn_params)

        return cnn_params, cnn_grads, self.local_sample_number, self.trainer.loss


    '''
    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.args)

        # test data
        test_metrics = self.trainer.test(self.test_local, self.args)
    '''
