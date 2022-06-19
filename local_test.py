import argparse
import logging
import os
import sys
import time

import numpy as np
import copy

import torch
import torch.nn as nn
#import torch_hd.hdlayers as hd
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchmetrics.functional import accuracy
import torchvision.transforms as transforms


from FedML.fedml_api.distributed.BaselineCNN.cnn_ModelTrainer import MyModelTrainer
#from FedML.fedml_api.distributed.BaselineCNN.cnn_Trainer import BaseCNN_Trainer
#from FedML.fedml_api.distributed.BaselineCNN.cnn_ClientManager import BaseCNNClientManager
from FedML.fedml_api.distributed.BaselineCNN.cnnAggregator import BaselineCNNAggregator
#from FedML.fedml_api.distributed.BaselineCNN.cnnServerManager import BaselineCNNServerManager

from FedML.fedml_api.data_preprocessing.load_data import load_partition_data
from FedML.fedml_api.data_preprocessing.load_data import load_partition_data_shakespeare
from FedML.fedml_api.data_preprocessing.load_data import load_partition_data_HAR
from FedML.fedml_api.data_preprocessing.load_data import load_partition_data_HPWREN


from FedML.fedml_api.model.Baseline.FashionMNIST import FashionMNIST_Net
from FedML.fedml_api.model.Baseline.MNIST import MNIST_Net
from FedML.fedml_api.model.Baseline.CIFAR10 import CIFAR10_Net
from FedML.fedml_api.model.Baseline.shakespeare import Shakespeare_Net
from FedML.fedml_api.model.Baseline.HAR import HAR_Net
from FedML.fedml_api.model.Baseline.HPWREN import HPWREN_Net


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from utils import *


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """

    parser.add_argument('--dataset', type=str, default='hpwren',
                        choices=['mnist', 'fashionmnist', 'cifar10', 'shakespeare','har','hpwren'],
                        help='dataset used for training')


    parser.add_argument('--partition_method', type=str, default='iid',
                        choices=['iid', 'noniid'],
                        help='how to partition the dataset on local clients')


    parser.add_argument('--client_num_in_total', type=int, default=6,
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=6,
                        help='number of workers')


    parser.add_argument('--batch_size', type=int, default=10,
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epochs', type=int, default=5,
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=20,
                        help='how many round of communications we should use')


    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    # Communication settings
    parser.add_argument('--backend', type=str, default='MQTT',
                        choices=['MQTT', 'MPI'],
                        help='communication backend')
                        
                        
    parser.add_argument('--mqtt_host', type=str, default='127.0.0.1',
                        help='host IP in MQTT')
                        
                        
    parser.add_argument('--mqtt_port', type=int, default=1883,
                        help='host port in MQTT')



    #
    parser.add_argument('--partition_alpha', type=float, default=0.5,
                        help='partition alpha (default: 0.5), used as the proportion'
                             'of majority labels in non-iid in latest implementation')

    parser.add_argument('--partition_secondary', type=bool, default=False,
                        help='True to sample minority labels from one random secondary class,'
                             'False to sample minority labels uniformly from the rest classes except the majority one')

    parser.add_argument('--partition_label', type=str, default='uniform',
                        choices=['uniform', 'normal'],
                        help='how to assign labels to clients in non-iid data distribution')

    parser.add_argument('--data_size_per_client', type=int, default=500,
                        help='input batch size for training (default: 64)')
                        
    
    
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='sgd optimizer momentum 0.9')

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='weight_decay (default: 5e-4)')

    parser.add_argument('--method', type=str, default="fedsync",
                         choices=['fedsync', 'fedasync'],
                         help='fedmethod')


    args = parser.parse_args()
    return args







def load_data(args, dataset_name):
    if dataset_name == "shakespeare":
        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size,"FedML/data/shakespeare")
        #args.client_num_in_total = len(train_data_local_dict)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))

    elif dataset_name == "har":
        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_HAR(args.batch_size,"FedML/data/HAR")
        #args.client_num_in_total = len(train_data_local_dict)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))


    elif dataset_name == "hpwren":
        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_HPWREN(args.batch_size,"FedML/data/HPWREN")
        #args.client_num_in_total = len(train_data_local_dict)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))


    elif dataset_name == "mnist" or dataset_name == "fashionmnist" or \
        dataset_name == "cifar10":
        data_loader = load_partition_data
        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        data_dir = './../data/' + args.dataset
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, data_dir, args.partition_method,
                                args.partition_label, args.partition_alpha, args.partition_secondary,
                                args.client_num_in_total, args.batch_size,
                                args.data_size_per_client)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))

    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model(args):
    if args.dataset == "mnist":
        model = MNIST_Net()
    elif args.dataset == "fashionmnist":
        model = FashionMNIST_Net()
    elif args.dataset == "cifar10":
        model = CIFAR10_Net()
    elif args.dataset == "shakespeare":
        model = Shakespeare_Net()
    elif args.dataset == "har":
        model = HAR_Net()
    elif args.dataset == "hpwren":
        model = HPWREN_Net()
    else:
        print("Invalid dataset")
        exit(0)
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    print(args)
    
    
    run_name = args.dataset + "_c_" + str(args.comm_round) + "_s_" + str(args.data_size_per_client) + "(cnn)"
    log_path = "hist/" + run_name
    
    print("Logging to: " + log_path)
    
    os.mkdir(log_path)
    
    parameters_file = log_path + "/Parameters.txt"
    logging_file = log_path + "/log.txt"
    graph_file = log_path + "/graph.jpg"

    parameters_log = open(parameters_file, "w")
    parameters_log.write(str(args))
    parameters_log.close()
    
    info_log = open(logging_file, "w")
    
    start_time = time.time()

    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset



    '''
    print("==================================")
    print("train_data_num: " , train_data_num)
    print("test_data_num: " , test_data_num)

    print("len(train_data_global): " , len(train_data_global))
    print("len(test_data_global): " , len(test_data_global))

    print("len(train_data_local_num_dict): ",len(train_data_local_num_dict))
    print("len(train_data_local_dict): ",len(train_data_local_dict))

    print("len(test_data_local_dict): ",len(test_data_local_dict))
    print("class_num: ", class_num)

    print("==================================")
    '''

    test_batch_selection = []
    for i in range(50):
        test_batch_selection.append(i)


    #model = create_model(args)
    
    #model = Linear_SimCLr()
    model = create_model(args)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    # models
    global_model = MyModelTrainer(model,args,device)
    global_model.set_id("Server")

    clients = []

    for i in range(args.client_num_in_total):
        clients.append(MyModelTrainer(model,args,device))
        clients[i].set_id(i)


    # wrap global model with aggregator
    global_model_aggregator = BaselineCNNAggregator(args,train_data_global, test_data_global, train_data_num,
                                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                  args.client_num_per_round, device, global_model)




    print("init complete, training start")


    
    acc_over_rounds=[]
    for c in range(args.comm_round):
        print("\n\n"+"Round#"+str(c))
        print("===================RoundStart====================")
        local_model = []
        for i in range(len(clients)):
            print("++++++++++++++++++TrainClient: "+str(i))
            clients[i].train(train_data_local_dict[i],args)

            local_model.append(clients[i].get_model_params())

        print("+++++++++++++++++++Aggregating")
        global_model_aggregator.local_test_aggregate(local_model)
        print("+++++++++++++++++++distribute")
        for i in range(len(clients)):
            clients[i].set_model_params(global_model.get_model_params())
        print("+++++++++++++++++++Test")
        if c % args.frequency_of_the_test == 0:
            test_loss, acc = global_model_aggregator.test_on_server_for_all_clients(c,test_batch_selection)
            print("Round acc: "+str(acc))
            print("Round loss: "+str(test_loss))
            acc_over_rounds.append(acc)
            info_log.write(str(c)+","+str(acc)+"\n")
        else:
            print("skipped")
        print("===================RoundComplete====================")
    


    duration = time.time()-start_time
    info_log.close()
    print("-----Processing Graph-----")
    
    
    df = pd.read_csv(logging_file, sep=",", header=None,names=["Round", "Acc"])
    graph = sns.lineplot(x = "Round", y = "Acc", data = df).get_figure()
    graph.savefig(graph_file) 
    
    
    print("Final Report:")
    print(acc_over_rounds)
    
    
    print("-----Finished-----")
    print("Training Time: ", duration)


    exit(0)





