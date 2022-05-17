from networks import regressor_circular
from data_utils import read_freq_data, SignalDataset
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch
import argparse
import json
import torch.optim as optim
# import tensorflow as tf
import numpy as np
import pandas as pd
import random as python_random
from datetime import datetime
import os
from torch.utils.data import random_split, DataLoader
import shutil
today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
from torchinfo import summary
#HRV Regressor model
import copy

class HrvRegressor:
    def __init__(self, config_path, signal_type="original"):
        # load data
        config=json.load(open(config_path, 'r'))
        self.config_path = config_path
        torch.manual_seed(1)
        np.random.seed(1)     
        python_random.seed(1)
        self.save_path = config["save_path"]
        input_signal, output_signal, self.labels = read_freq_data(config["folder_path"],  config["signal_percentage"])
        full_dataset = SignalDataset(input_signal, output_signal, self.labels, transforms.ToTensor())
        train_split, val_split, test_split = random_split(full_dataset, [650, 194, 195])
        ann_upsampler = torch.load(config["upsampler_path"])
        hrv_data = pd.read_csv(config["hrv_data"], header=None)
        hrv_data[0].fillna(value=hrv_data[0].mean(), inplace=True)
        hrv_data = hrv_data.to_numpy()
        if signal_type == "reconstructed":
            get_input = lambda x: torch.from_numpy(np.array([ann_upsampler(inp.float()).detach().numpy() for inp, op, label in x]))
        else:
            get_input = lambda x: torch.from_numpy(np.array([op.float().numpy() for inp, op, label in x]))
        get_label = lambda x: torch.from_numpy(np.array([hrv_data[i] for i in x.indices]))
        self.train_input, self.train_labels = get_input(train_split), get_label(train_split)
        self.train_set = list(zip(self.train_input, self.train_labels))
        self.val_input, self.val_labels = get_input(val_split), get_label(val_split)
        
        self.val_set = list(zip(self.val_input, self.val_labels))
        self.test_input, self.test_labels = get_input(test_split), get_label(test_split)
        self.test_set = list(zip(self.test_input, self.test_labels))
        
        self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size = config["batch_size"])
        self.val_loader = DataLoader(self.val_set, shuffle=True, batch_size = config["batch_size"])
        self.test_loader = DataLoader(self.test_set, batch_size = len(self.test_set))

        # Loss Function
        self.loss_func = torch.nn.MSELoss()
        self.regressor = regressor_circular.CirConvNet(config["layer_sizes"], config["dropout"])
        # summary(self.regressor, (1, 56), device='cpu')
        # for name, param in self.regressor.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape)
        print(list(self.regressor.parameters()))
        pytorch_total_params = sum(p.numel() for p in self.regressor.parameters() if p.requires_grad)
        print(pytorch_total_params)
        self.optimizer = optim.Adam(self.regressor.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        # self.optimizer = optim.SGD(self.regressor.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.epochs = config["epochs"]
        self.save_path = config["save_path"]
        self.best_model = None
    
    def accuracy(self, loader, model=None):
        if model is None:
            model = self.regressor
        acc = 0.0
        with torch.no_grad():
            for inp, labels in loader:
                pred = model(inp.float())
                loss = self.loss_func(pred, labels.float())
                acc += loss.item()
        return acc/len(loader)
    
    
    def train(self):
        min_mse = 1000000
        best_val_epoch = -1
        best_val_mse = -1
        for epoch in range(0, self.epochs):
            train_loss = 0.0
            for inp, label in self.train_loader:
                self.optimizer.zero_grad()
                pred = self.regressor(inp.float())
                l1_regularization = 0
                for param in self.regressor.parameters():
                        l1_regularization += param.abs().sum()
                loss = self.loss_func(pred, label.float()) # + 0.5*l1_regularization
                l1_regularization = 0.

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            val_loss = 0.0
            val_accuracy = self.accuracy(self.val_loader) 
            if val_accuracy < min_mse:
                min_mse = val_accuracy
                self.best_model = copy.deepcopy(self.regressor)
                best_val_epoch = epoch
                best_val_mse = min_mse
            if not epoch % 100:
                print(f"Epoch - {epoch}, train loss - {train_loss/len(self.train_loader)*1.0:.5f}, Train accuracy - {self.accuracy(self.train_loader)*1.0:.5f}, val accuracy - {val_accuracy*1.0:.5f}")
            
            
        with torch.no_grad():
            model_info_path = os.path.join(self.save_path, str(today))
            if os.path.exists(model_info_path):
                shutil.rmtree(model_info_path)
            os.makedirs(model_info_path)
            torch.save(self.regressor, os.path.join(model_info_path, f"regressor_{today}.pt"))
            shutil.copyfile(self.config_path, os.path.join(model_info_path, "config.json"))
        print(best_val_epoch, best_val_mse)
        print(f"Test Accuracy - {self.accuracy(self.test_loader, self.best_model)}")
        print(today)
    def load_model(self, model_weights_path):
        self.regressor_net.load_weights(model_weights_path)
    
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='HRV Regressor Training')
    parser.add_argument('--config', help="path to config with training params", default="config.json")
    parser.add_argument('--signal_type', help="Signal type", default="original", choices=["original", "reconstructed"])
    args = parser.parse_args()
    trainer =  HrvRegressor(config_path=args.config, signal_type=args.signal_type)
    trainer.train()