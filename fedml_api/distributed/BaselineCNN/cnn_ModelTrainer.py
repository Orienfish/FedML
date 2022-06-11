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

    # set cnn parameter
    def set_model_params(self, model_parameters):
        self.classifier.load_state_dict(model_parameters)

    # train
    def train(self, train_data, args):
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
                logging.info('Epoch: [{}/{}]\tLoss: {:.6f}\t'
                      'L2 Loss: {:.6f}'.format(epoch, args.epochs,
                                               sum(epoch_loss) / len(epoch_loss),
                                               sum(epoch_l2_loss) / len(epoch_l2_loss)))
            else:
                logging.info('Epoch: [{}/{}]\tLoss: {:.6f}'.format(epoch, args.epochs,
                                                            sum(epoch_loss) / len(epoch_loss)))
            self.loss = sum(epoch_loss) / len(epoch_loss)

    
    # test
    def test(self, test_data, args, batch_selection):
        
        model = self.classifier.to(self.device)
        self.classifier.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)

        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(test_data):
            if batch_selection is not None and batch_idx not in batch_selection:
                continue

            x, y = x.to(self.device), y.to(self.device)
            outputs = model(x)

            test_loss += criterion(outputs, y).item()

            _, y_hat = outputs.max(1)
            correct += y_hat.eq(y).sum().item()
            total += y.size(0)

        test_loss /= len(test_data)
        acc = correct / total

        return test_loss, acc


    # not used
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
