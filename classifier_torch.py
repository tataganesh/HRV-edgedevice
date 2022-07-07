import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import torch
import torch.nn as nn
import os
from data_utils import read_freq_data, get_all_sets
from torch.utils.data import random_split
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
from joblib import dump
from datetime import datetime
import pandas as pd
import json
import random as python_random
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from networks import classifier_circular
import copy
today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
import torch.nn.functional as F
from torchinfo import summary
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable

# class FocalLoss(nn.modules.loss._WeightedLoss):
#     def __init__(self, weight=None, gamma=2,reduction='mean'):
#         super(FocalLoss, self).__init__(weight,reduction=reduction)
#         self.gamma = gamma
#         self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

#     def forward(self, input, target):
#         ce_loss = F.binary_cross_entropy(torch.sigmoid(input), target,reduction=self.reduction,weight=self.weight)
#         pt = torch.exp(-ce_loss)
#         focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
#         return focal_loss
activations = dict()
def conv_inp_op(self, inp, outp):
    activations["input"] = inp[0].detach()
    activations["output"] = outp.detach()

class AnomalyClassifier:
    def __init__(self, config_path):
        # load data
        config=json.load(open(config_path, 'r'))
        self.config_path = config_path
        torch.manual_seed(1)
        np.random.seed(1)     
        python_random.seed(1)
        self.save_path = config["save_path"]
        self.upsampler_layer_sizes = config["upsampler_layer_sizes"]
        input_signal, output_signal, self.labels = read_freq_data(config["folder_path"])
        train_split, val_split, test_split = get_all_sets(input_signal, output_signal, self.labels)
        ann_upsampler = torch.load(config["upsampler_path"])
        get_input = lambda x: torch.from_numpy(np.array([ann_upsampler(inp.float()).detach().numpy() for inp, op, label in x]))
        get_label = lambda x: np.array([label for inp, op, label in x])
        self.train_input, self.train_labels = get_input(train_split), get_label(train_split)
        self.train_set = list(zip(self.train_input, self.train_labels))
        self.val_input, self.val_labels = get_input(val_split), get_label(val_split)
        
        self.val_set = list(zip(self.val_input, self.val_labels))
        self.test_input, self.test_labels = get_input(test_split), get_label(test_split)
        self.test_set = list(zip(self.test_input, self.test_labels))
        
        print(f"Train Normal Signals - {np.sum(self.train_labels==0)}")
        print(f"Train Abnormal Signals - {np.sum(self.train_labels==1)}")

        print(f"Test Normal Signals - {np.sum(self.test_labels==0)}")
        print(f"Test Abnormal Signals - {np.sum(self.test_labels==1)}")
        
        self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size = config["batch_size"])
        self.val_loader = DataLoader(self.val_set, shuffle=True, batch_size = config["batch_size"])
        self.test_loader = DataLoader(self.test_set, batch_size = len(self.test_set))


        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # Loss Function
        print(torch.tensor(np.sum(self.train_labels==0)/np.sum(self.train_labels==1)))
        if config["pos_weight"]:
            self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sum(self.train_labels==0)/np.sum(self.train_labels==1)))
        else:
            self.loss_func = torch.nn.BCEWithLogitsLoss()
        if config["conv2d"]:
            self.classifier = classifier_circular.CirConvNet()
        else:
            self.classifier = classifier_circular.CirConvNetStacked1d(config["layer_sizes"])
        summary(self.classifier, (1, 69), device='cpu')
        self.classifier = self.classifier.to(self.device)

        print(list(self.classifier.parameters()))
        
        # Register hook
        # self.classifier.conv1.register_forward_hook(conv_inp_op)
        pytorch_total_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        print(pytorch_total_params)
        # self.optimizer = optim.Adam(self.regressor.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.optimizer = optim.SGD(self.classifier.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.epochs = config["epochs"]
        self.save_path = config["save_path"]
        self.best_model = None
        self.save_model = config["save_model"]
        

    def accuracy(self, loader, model=None, show_confusion=False, validation=False, loader_labels=None):
        if model is None:
            model = self.classifier
        acc = 0.0
        normal_acc = 0
        abnormal_acc = 0
        self.predictions = list()
        with torch.no_grad():
            for inp, labels in loader:
                inp = inp.to(self.device)
                labels = labels.to(self.device)
                pred = model(inp.float())
                pred[pred > 0.5] = 1
                pred[pred < 0.5] = 0
                acc += torch.sum(pred.squeeze(1) == labels)
                normal_acc += torch.sum(pred.squeeze(1)[labels == 0] == 0)
                abnormal_acc += torch.sum(pred.squeeze(1)[labels == 1])
        if show_confusion:
            pred = model(self.test_input).detach()
            pred[pred > 0.5] = 1
            pred[pred < 0.5] = 0
            cm = confusion_matrix(self.test_labels, pred.squeeze(1))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["Normal", 'Abnormal'])
            disp.plot()
            plt.show()
            return acc/len(loader.dataset), abnormal_acc/np.sum(loader_labels==1), normal_acc/np.sum(loader_labels == 0), f1_score(loader_labels, pred)

        if validation:
            pred = model(self.val_input).detach().squeeze(1)
            pred[pred > 0.5] = 1
            pred[pred < 0.5] = 0
            try:
                return acc/len(loader.dataset), abnormal_acc/np.sum(loader_labels==1), normal_acc/np.sum(loader_labels == 0), f1_score(loader_labels, pred)
            except:
                return acc/len(loader.dataset), abnormal_acc/np.sum(loader_labels==1), normal_acc/np.sum(loader_labels == 0), 0.0
        else:
            return acc/len(loader.dataset), abnormal_acc/np.sum(loader_labels==1), normal_acc/np.sum(loader_labels == 0), 0

            
        
    def train(self):
        # print(f"Test Accuracy - {self.accuracy(self.test_loader, self.classifier, show_confusion=True)}")
        best_val_acc = -1
        best_model = None
        best_epoch = None
        for epoch in range(0, self.epochs):
            train_loss = 0.0
            for inp, label in self.train_loader:
                self.optimizer.zero_grad()
                pred = self.classifier(inp.float())
                loss = self.loss_func(pred.squeeze(1), label.float()) # + 0.5*l1_regularization
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            val_loss = 0.0
            val_accuracy, val_adnormal, val_normal, f1_val = self.accuracy(self.val_loader, validation=True, loader_labels=self.val_labels) 
            train_accuracy, train_adnormal, train_normal, _ = self.accuracy(self.train_loader, loader_labels=self.train_labels) 
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model  = copy.deepcopy(self.classifier)
                best_epoch = epoch
            if not epoch % 100:
                print(f"Epoch - {epoch}, train loss - {train_loss/len(self.train_loader)*1.0:.5f}, Train accuracy - {train_accuracy:.5f}, val accuracy - {val_accuracy:.5f}, val abnormal - {val_adnormal}, val normal - {val_normal}")
            
        if self.save_model:
            print("Saving Model")
            with torch.no_grad():
                model_info_path = os.path.join(self.save_path, str(today))
                if os.path.exists(model_info_path):
                    shutil.rmtree(model_info_path)
                os.makedirs(model_info_path)
                torch.save(self.classifier, os.path.join(model_info_path, f"classifier_{today}.pt"))
                shutil.copyfile(self.config_path, os.path.join(model_info_path, "config.json"))
        print(best_epoch, val_accuracy)
        print(f"Test Accuracy - {self.accuracy(self.test_loader, best_model, show_confusion=True, loader_labels=self.test_labels)}")
        # fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
        # axs[0].plot(np.arange(56), self.test_set[0][0])

        # axs[1].imshow(activations["input"][0][0].permute(1, 0))
        # axs[2].imshow(torch.relu(activations["output"][0][0].permute(1,0)))
        # plt.show()
        print(today)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HRV Regressor Training')
    parser.add_argument('--config', help="path to config with training params", default="config.json")
    parser.add_argument('--signal_type', help="Signal type", default="original", choices=["original", "reconstructed"])
    args = parser.parse_args()
    trainer =  AnomalyClassifier(config_path=args.config)
    trainer.train()
