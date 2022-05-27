import logging

import torch
from torch import nn
import torch.optim as optim
import copy
from torchmetrics.functional import accuracy

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer




class MyModelTrainer(ModelTrainer):
    
    # assign initial model and encoder
    def __init__(self, model,args,device):
        self.classifier = copy.deepcopy(model)
        self.round = 0;
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

        model = self.classifier.to(self.device)
        self.classifier.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        criterion = nn.CrossEntropyLoss().to(self.device)

        epoch_loss = []
        
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                #print("TRAINING------------------", batch_idx)
                x = x.to(self.device)
                labels = labels.to(self.device).type(torch.long)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print('(Trainer_ID {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.id,epoch,sum(epoch_loss) / len(epoch_loss)))



        
    
    # test
    def test(self, test_data, args, batch_selection):
        
        model = self.classifier.to(self.device)
        self.classifier.eval()
        
        acc = 0
        batch_correct = []
        
        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                if batch_selection!=None and batch_idx not in batch_selection:
                    continue
                x = x.to(self.device)
                target = target.to(self.device)
                
                #print("Testing: ",batch_idx)
                pred = model(x)
                
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                
                #print(target)
                #print(predicted)
                
                batch_correct.append(predicted.eq(target).sum())
            acc = sum(batch_correct) / (len(batch_selection)*args.batch_size)
        print("-----------------------------Test acc: ", str(acc)) 

        return acc




        

    # not used
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
