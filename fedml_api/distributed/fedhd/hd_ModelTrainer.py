import logging

import torch
from torch import nn
import torch.optim as optim
import copy
import numpy as np
import pickle

from encoder import *
from pl_bolts.models.self_supervised import SimCLR

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    
    # assign initial model and encoder
    def __init__(self, args, device):
        self.round = 0
        self.device = device

        self.n_classes = 10
        self.class_hvs_best = np.zeros((self.n_classes, args.D))
        self.acc_max = -1

        # load base matrix / vector
        if args.dataset == "mnist" or args.dataset == "fashionmnist":
            with open('base/base_matrix_fmnist_mnist_6_26.hd', 'rb') as file:
                self.base_matrix = pickle.load(file)
            with open('base/base_vector_fmnist_mnist_6_26.hd', 'rb') as file:
                self.base_vector = pickle.load(file)
            print("28x28 loaded")
        elif args.dataset == "har":
            with open('base/base_matrix_har_6_26.hd', 'rb') as file:
                self.base_matrix = pickle.load(file)
            with open('base/base_vector_har_6_26.hd', 'rb') as file:
                self.base_vector = pickle.load(file)
            print("561 loaded")
        elif args.dataset == "cifar10":
            with open('base/base_matrix_cifar10_6_26.hd', 'rb') as file:
                self.base_matrix = pickle.load(file)
            with open('base/base_vector_cifar10_6_26.hd', 'rb') as file:
                self.base_vector = pickle.load(file)
            print("2048 loaded")
            self.cifar_feature_extractor = SimCLR.load_from_checkpoint(
                "epoch=960.ckpt", strict=False, dataset='imagenet', maxpool1=False, first_conv=False, input_height=28)
            self.cifar_feature_extractor.freeze()
            print("SimCLr module loaded")
        else:
            print("Unimplemented")
            exit(1)


    # get cnn parameters
    def get_model_params(self):
        return self.class_hvs_best

    # set cnn parameters
    def set_model_params(self, model_parameters):
        self.class_hvs_best = deepcopy(model_parameters)

    def get_id(self):
        return self.id

    # train
    def train(self, train_data,test_data, args):
        self.acc_max = -1
        base_matrix = self.base_matrix
        base_vector = self.base_vector
        for epoch in range(args.epochs):
            class_hvs = deepcopy(self.class_hvs_best)
            for batch_idx, (x, y) in enumerate(train_data):
                # encode x
                if args.dataset == "cifar10":
                    x = self.cifar_feature_extractor(x)
                train_enc_hvs = encoding_nonlinear(x, base_matrix, base_vector)
                pickList = np.arange(0, len(train_enc_hvs))
                np.random.shuffle(pickList)
                for i in pickList:
                    predict = max_match_nonlinear(class_hvs, train_enc_hvs[i])
                    if predict != y[i]:
                        class_hvs[predict] = class_hvs[predict] - args.rate*(train_enc_hvs[i].numpy())
                        class_hvs[y[i]]    = class_hvs[y[i]] + args.rate*(train_enc_hvs[i].numpy())
        self.class_hvs_best = deepcopy(class_hvs) # comment out is doing multiple epochs

    '''
            correct = 0
            for batch_idx, (x, y) in enumerate(test_data):
                # encode x
                if batch_idx >= (len(test_data) / args.batch_size):
                    break
                test_enc_hvs = encoding_nonlinear(x, base_matrix, base_vector)
                for i in range(len(test_enc_hvs)):
                    predict = max_match_nonlinear(class_hvs, test_enc_hvs[i])
                    if predict == y[i]:
                        correct+=1

            print(correct)
            acc = correct/len(test_data)
            print("[ Epoch {e} Acc {a} ]".format(e = epoch, a = acc))
            if acc > self.acc_max:
                print("Updating")
                self.class_hvs_best = deepcopy(class_hvs)
                self.acc_max = acc
            else:
                print("No update")
    '''

    #test
    def test(self, test_data, args, batch_selection):
        base_matrix = self.base_matrix
        base_vector = self.base_vector
        correct = 0
        tested = 0
        for batch_idx, (x, y) in enumerate(test_data):
            if batch_idx not in batch_selection:
                continue
            if args.dataset == "cifar10":
                x = self.cifar_feature_extractor(x)
            test_enc_hvs = encoding_nonlinear(x, base_matrix, base_vector)
            for i in range(len(x)):
                predict = max_match_nonlinear(self.class_hvs_best, test_enc_hvs[i])
                tested+=1
                if predict==y[i]:
                    correct+=1

        test_acc = correct/tested

        return 0, test_acc

