import copy
import logging
import time

import numpy as np
import wandb

import torch

from .utils import transform_list_to_tensor


class BaselineCNNAggregator(object):

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
        logging.info("Receive model index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num

    def aggregate(self, flag_client_model_uploaded):
        model_list = []

        # Partly received
        local_sample_number = self.args.data_size_per_client
        training_num = 0
        for idx in range(self.worker_num):
            if flag_client_model_uploaded[idx]:
                model_list.append(self.model_dict[idx])
                training_num += local_sample_number
        w = local_sample_number / training_num
        
        averaged_params = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_model_params = model_list[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
                    
        self.set_global_model_params(averaged_params)

        return averaged_params

    def aggregate_async(self, model_params, sample_num, staleness):
        alpha_t = self.args.alpha * self.staleness(staleness)
        logging.info('{} alpha: {} staleness: {} alpha_t: {}'.format(
            self.args.staleness_func, self.args.alpha, staleness, alpha_t
        ))

        global_model_params = self.get_global_model_params()
        averaged_params = copy.deepcopy(global_model_params)
        for k in averaged_params.keys():
            averaged_params[k] = (1 - alpha_t) * global_model_params[k] + \
                alpha_t * model_params[k]

        self.set_global_model_params(averaged_params)

        # print("Averaged")
        # print(type(averaged_params))
        return averaged_params

    def staleness(self, staleness):
        if self.args.staleness_func == "constant":
            return 1
        elif self.args.staleness_func == "polynomial":
            a = 0.5
            return pow(staleness+1, -a)
        elif self.args.staleness_func == "hinge":
            a, b = 10, 4
            if staleness <= b:
                return 1
            else:
                return 1 / (a * (staleness - b) + 1)

    '''
    # for local test only
    def aggregate(self,client_model_list):
        model_list = []
        
        local_sample_number = self.args.data_size_per_client
        training_num = (self.args.client_num_per_round) * local_sample_number
        w = local_sample_number / training_num
        
        for idx in range(self.worker_num):
            model_list.append(client_model_list[idx])
        
        averaged_params = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_model_params = model_list[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        
        self.set_global_model_params(averaged_params)

        print("Averaged")
        return averaged_params
    '''


    def test_on_server_for_all_clients(self, round_idx, batch_selection=None):
        accuracy = self.classifier.test(self.test_global, self.args, batch_selection)
        return accuracy















