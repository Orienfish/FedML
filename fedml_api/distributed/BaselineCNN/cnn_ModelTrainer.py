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

    # get cnn parameters
    def get_model_params(self):
        return self.classifier.cpu().state_dict()

    # set snn parameter
    def set_model_params(self, model_parameters):
        self.classifier.load_state_dict(model_parameters)







    # train
    def train(self, train_data, args):
        if args.dataset == "shakespeare":
            print("Train func: train_shakespeare")
            self.train_shakespeare(train_data, args)
        else:
            print("Train func: train_cifar10_mnist_fashionmnist")
            self.train_general(train_data, args)


    def train_general(self, train_data, args):
        old_model = copy.deepcopy(self.classifier)
        old_model.to(self.device)
        old_model.eval()

        model = self.classifier.to(self.device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        criterion = nn.CrossEntropyLoss().to(self.device)

        epoch_loss, epoch_l2_loss = [], []
        for epoch in range(args.epochs):
            batch_loss, batch_l2_loss = [], []
            for batch_idx, (x, y) in enumerate(train_data):
                x, y = x.to(self.device), y.to(self.device).type(torch.long)
                optimizer.zero_grad()
                outputs = model(x)
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


    def train_shakespeare(self,train_data,args):

        # train config
        reg=True
        rou=None

        old_model = copy.deepcopy(self.classifier)
        old_model.to(self.device)
        old_model.eval()

        self.classifier.to(self.device)
        self.classifier.train()
        criterion = nn.CrossEntropyLoss().to(self.device)


        optimizer = optim.SGD(self.classifier.parameters(), lr=args.lr, momentum=args.momentum)


        batch_size, state_h, state_c = None, None, None

        train_loss, train_l2_loss = 0, 0
        correct = 0

        for epoch in range(args.epochs):
            for (batch_idx, (image, label)) in enumerate(train_data):

                image, label = image.to(self.device), label.to(self.device).type(torch.long)
                optimizer.zero_grad()

                if batch_size is None:
                    batch_size = image.shape[0]
                    state_h, state_c = self.classifier.zero_state(batch_size)
                    state_h = state_h.to(self.device)
                    state_c = state_c.to(self.device)

                if image.shape[0] < batch_size:  # Less than one batch
                    break

                output, (state_h, state_c) = self.classifier(image, (state_h, state_c))

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss = criterion(output, label)

                # Add regularization
                if reg is not None and rou is not None:
                    l2_loss = 0.0
                    for paramA, paramB in zip(self.classifier.parameters(),
                                              old_model.parameters()):
                        l2_loss += rou / 2 * \
                                   torch.sum(torch.square(paramA - paramB.detach()))
                    loss += l2_loss
                    train_l2_loss += l2_loss.item()

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                #logging.info('Epoch: [{}/{}]\tLoss: {:.6f}'.format(epoch, args.epochs, loss.item()))


                _, predicted = output.max(1)
                # print(predicted)
                correct += predicted.eq(label).sum().item()

        correct /= args.epochs
        total = len(train_data)  # Total # of train samples
        train_loss = train_loss / ((batch_idx)*args.epochs)
        accuracy = correct / total

        print('Train accuracy: '+ str(accuracy))

        if reg is not None and rou is not None:
            train_l2_loss = train_l2_loss / ((batch_idx)*args.epochs)
            logging.info(
                'loss: {} l2_loss: {}'.format(train_loss, train_l2_loss))
        else:
            logging.info(
                'loss: {}'.format(train_loss))

        self.loss = train_loss










    
    # test
    def test(self, test_data, args, batch_selection):
        if args.dataset == "cifar10" or args.dataset == "mnist" or args.dataset == "fashionmnist":
            print("Test func: test_cifar10_mnist_fashionmnist")
            return(self.test_cifar10_mnist_fashionmnist(test_data, args, batch_selection))
        elif args.dataset == "shakespeare":
            print("Test func: test_shakespeare")
            return(self.test_shakespeare(test_data, args, batch_selection))
        else:
            print("Unimplemented")

    def test_cifar10_mnist_fashionmnist(self, test_data, args, batch_selection):
        
        model = self.classifier.to(self.device)
        self.classifier.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)

        test_loss = 0
        correct = 0

        total = args.batch_size * len(batch_selection)
        for batch_idx, (x, y) in enumerate(test_data):
            if batch_selection is not None and batch_idx not in batch_selection:
                continue

            x, y = x.to(self.device), y.to(self.device).type(torch.long)
            outputs = model(x)

            test_loss += criterion(outputs, y).item()

            _, y_hat = outputs.max(1)


            correct += y_hat.eq(y).sum().item()

        test_loss /= len(batch_selection)
        acc = correct / total

        return test_loss, acc



    def test_shakespeare(self,test_data,args,batch_selection):
        self.classifier.to(self.device)
        self.classifier.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)

        test_loss = 0
        correct = 0
        total = args.batch_size * len(batch_selection)

        batch_size, state_h, state_c = None, None, None

        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(test_data):
                if batch_idx not in batch_selection:
                    continue

                image, label = image.to(self.device), label.to(self.device).type(torch.long)

                if batch_size is None:
                    batch_size = image.shape[0]
                    state_h, state_c = self.classifier.zero_state(batch_size)
                    state_h = state_h.to(self.device)
                    state_c = state_c.to(self.device)

                if image.shape[0] < batch_size:  # Less than one batch
                    break

                output, (state_h, state_c) = self.classifier(image, (state_h, state_c))

                state_h = state_h.detach()
                state_c = state_c.detach()

                # sum up batch loss
                test_loss += criterion(output, label).item()
                # get the index of the max log-probability
                _, predicted = output.max(1)
                correct += predicted.eq(label).sum().item()

        test_loss = test_loss / len(batch_selection)
        accuracy = correct / total

        return test_loss, accuracy




    # not used
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
