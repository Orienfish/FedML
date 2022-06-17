import logging

import torch
from torch import nn
import torch.optim as optim
import copy

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    
    # assign initial model and encoder
    def __init__(self, model, args, device):
        self.classifier = copy.deepcopy(model)
        self.round = 0
        self.device = device
        self.partition_method = args.partition_method
        self.init_trainer(args)

    # trainer init
    def init_trainer(self,args):
        model = self.classifier
        if args.dataset == "mnist":
            print("Train MNIST")
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif args.dataset == "fashionmnist":
            print("Train FashionMNIST")
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif args.dataset == "cifar10":
            print("Train CIFAR10")
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif args.dataset == "shakespeare":
            print("Train shakesprare")
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif args.dataset == "har":
            print("Train HAR")
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif args.dataset == "hpwren":
            print("Train HPWREN")
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            self.criterion = nn.MSELoss().to(self.device)
        else:
            print("Unimplemented")


    # get cnn parameters
    def get_model_params(self):
        return self.classifier.cpu().state_dict()

    # set snn parameter
    def set_model_params(self, model_parameters):
        self.classifier.load_state_dict(model_parameters)



    # train
    def train(self, train_data, args):
        old_model = copy.deepcopy(self.classifier)
        old_model.to(self.device)
        old_model.eval()

        model = self.classifier.to(self.device)
        model.train()

        optimizer = self.optimizer
        criterion = self.criterion

        if args.dataset == "shakespeare":
            batch_size, state_h, state_c = None, None, None

        epoch_loss, epoch_l2_loss = [], []
        for epoch in range(args.epochs):
            batch_loss, batch_l2_loss = [], []
            for batch_idx, (x, y) in enumerate(train_data):

                if args.dataset == "har":
                    x, y = x.to(self.device).type(torch.float), y.to(self.device).type(torch.long)
                    y = y - 1 # original label 1-6, suppose to be 1-5
                    outputs = model(x)
                elif args.dataset == "hpwren":
                    x, y = x.to(self.device).type(torch.float), y.to(self.device).type(torch.float)
                    hidden_size = 128   # need to match model file
                    num_layers = 1      # need to match model file
                    h_0 = torch.zeros(num_layers, x.shape[0], hidden_size).to(self.device)
                    c_0 = torch.zeros(num_layers, x.shape[0], hidden_size).to(self.device)
                    outputs = model(x,(h_0,c_0))
                elif args.dataset == "shakespeare":
                    x, y = x.to(self.device), y.to(self.device).type(torch.long)
                    if batch_size is None:
                        batch_size = x.shape[0]
                        state_h, state_c = model.zero_state(batch_size)
                        state_h = state_h.to(self.device)
                        state_c = state_c.to(self.device)
                    if x.shape[0] < batch_size:  # Less than one batch
                        break
                    outputs, (state_h, state_c) = model(x, (state_h, state_c))
                    state_h = state_h.detach()
                    state_c = state_c.detach()
                else:
                    x, y = x.to(self.device), y.to(self.device).type(torch.long)
                    outputs = model(x)

                optimizer.zero_grad()
                loss = criterion(outputs, y)

                # Add regularization
                if args.method == 'fedasync':
                    l2_loss = 0.0
                    for paramA, paramB in zip(model.parameters(),
                                              old_model.parameters()):
                        l2_loss += args.rou / 2 * \
                                   torch.sum(torch.square(paramA - paramB.detach()))
                    loss += l2_loss
                    batch_l2_loss.append(l2_loss.item())

                batch_loss.append(loss.item())

                loss.backward()
                optimizer.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if len(batch_l2_loss) > 0:
                epoch_l2_loss.append(sum(batch_l2_loss) / len(batch_l2_loss))
                print('Epoch: [{}/{}]\tLoss: {:.6f}\t'
                      'L2 Loss: {:.6f}'.format(epoch, args.epochs,
                                               sum(epoch_loss) / len(epoch_loss),
                                               sum(epoch_l2_loss) / len(epoch_l2_loss)))
            else:
                print('Epoch: [{}/{}]\tLoss: {:.6f}'.format(epoch, args.epochs,
                                                                sum(epoch_loss) / len(epoch_loss)))
            self.loss = sum(epoch_loss) / len(epoch_loss)


    #test
    def test(self, test_data, args, batch_selection):
        
        model = self.classifier.to(self.device)
        self.classifier.eval()
        criterion = self.criterion

        test_loss = 0
        correct = 0

        if args.dataset == "shakespeare":
            batch_size, state_h, state_c = None, None, None

        total = args.batch_size * len(batch_selection)
        for batch_idx, (x, y) in enumerate(test_data):
            if batch_selection is not None and batch_idx not in batch_selection:
                continue

            if args.dataset == "har":
                x, y = x.to(self.device).type(torch.float), y.to(self.device).type(torch.long)
                y = y - 1 # original label 1-6, suppose to be 1-5
                outputs = model(x)
            elif args.dataset == "hpwren":
                x, y = x.to(self.device).type(torch.float), y.to(self.device).type(torch.float)
                hidden_size = 128   # need to match model file
                num_layers = 1      # need to match model file
                h_0 = torch.zeros(num_layers, x.shape[0], hidden_size).to(self.device)
                c_0 = torch.zeros(num_layers, x.shape[0], hidden_size).to(self.device)
                outputs = model(x,(h_0,c_0))
            elif args.dataset == "shakespeare":
                x, y = x.to(self.device), y.to(self.device).type(torch.long)
                if batch_size is None:
                    batch_size = x.shape[0]
                    state_h, state_c = model.zero_state(batch_size)
                    state_h = state_h.to(self.device)
                    state_c = state_c.to(self.device)
                if x.shape[0] < batch_size:  # Less than one batch
                    break
                outputs, (state_h, state_c) = model(x, (state_h, state_c))
                state_h = state_h.detach()
                state_c = state_c.detach()
            else:
                x, y = x.to(self.device), y.to(self.device).type(torch.long)
                outputs = model(x)


            if args.dataset == "hpwren":
                test_loss += y.size(0) * criterion(outputs, y).item()
            else:
                test_loss += criterion(outputs, y).item()
                _, y_hat = outputs.max(1)
                correct += y_hat.eq(y).sum().item()

        test_loss /= len(batch_selection)
        acc = correct / total

        return test_loss, acc




    # not used
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
