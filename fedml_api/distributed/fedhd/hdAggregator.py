import copy
import logging
import time

import numpy as np
import wandb

import torch

from .utils import transform_list_to_tensor


class BaselineHDAggregator(object):

    def __init__(self, args, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                 total_num, device, model_trainer):

        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.classifier = model_trainer
        self.worker_num = total_num
        self.device = device
        self.args = args

        self.model_dict = dict()
        self.sample_num_dict = dict()

    def get_global_model_params(self):
        return self.classifier.get_model_params()
    
    def set_global_model_params(self, model_parameters):
        self.classifier.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num


    def aggregate(self):
        n_classes = 10
        update = np.zeros((n_classes, self.args.D))

        num = len(self.model_dict)

        for model_idx in self.model_dict:
            update += self.model_dict[model_idx]

        update = update / num

        self.set_global_model_params(update)

        return update


    def test_on_server_for_all_clients(self, round_idx, batch_selection=None):
        test_loss,accuracy = self.classifier.test(self.test_global, self.args, batch_selection)
        return test_loss,accuracy















