from sklearn.metrics import nan_euclidean_distances
from networks import regressor_circular
from data_utils import read_freq_data, SignalDataset, get_all_sets
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
import torch.nn.functional as F
import os
from torch.utils.data import random_split, DataLoader
import shutil
today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
from torchinfo import summary
import copy
import math
import matplotlib.pyplot as plt

class HrvRegressor:
    def __init__(self, config_path, signal_type="original"):
        # load data
        config = json.load(open(config_path, 'r'))
        self.config_path = config_path
        torch.manual_seed(config["random_seed"])
        np.random.seed(config["random_seed"])     
        python_random.seed(config["random_seed"])
        self.save_path = config["save_path"]
        self.save_model = config["save_model"]
        input_signal, output_signal, self.labels = read_freq_data(config["folder_path"], include_abnormal=True)
        hrv_data_normal_signals = pd.read_csv(config["hrv_data"], header=None).transpose().to_numpy()
        hrv_data = np.zeros(input_signal.shape[0])
        hrv_data[:hrv_data_normal_signals.shape[0]] = hrv_data_normal_signals[:, 0].astype(np.float32)
        train_split, val_split, test_split = get_all_sets(input_signal, output_signal, self.labels, hrv_data)
        ann_upsampler = torch.load(config["upsampler_path"])
        if signal_type == "reconstructed":
            get_dataset = lambda x: torch.from_numpy(np.array([ann_upsampler(inp.float()).detach().numpy() for inp, op, label, hrv in x if not label]))
        else:
            get_dataset = lambda x: torch.from_numpy(np.array([op.float().numpy() for inp, op, label, hrv in x if not label]))
        get_label = lambda x: np.array([hrv for inp, op, label, hrv in x if not label])
        
        # Datasets
        self.train_input, self.train_labels = get_dataset(train_split), get_label(train_split)
        self.train_set = list(zip(self.train_input, self.train_labels))
        
        self.val_input, self.val_labels = get_dataset(val_split), get_label(val_split)
        self.val_set = list(zip(self.val_input, self.val_labels))
        
        self.test_input, self.test_labels = get_dataset(test_split), get_label(test_split)
        self.test_set = list(zip(self.test_input, self.test_labels))
        
        print(f"Train Set Size: {len(self.train_set)}")
        print(f"Val Set Size: {len(self.val_set)}")
        print(f"Test Set Size: {len(self.test_set)}")
        
        # Dataloaders
        self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size = config["batch_size"])
        self.val_loader = DataLoader(self.val_set, batch_size = config["batch_size"])
        self.test_loader = DataLoader(self.test_set, batch_size = len(self.test_set))
        
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # Loss Function
        self.loss_func = torch.nn.MSELoss()
        if config["network_type"] == "fully_connected":
            self.regressor = regressor_circular.HRNet(2, 8, 69, 1).to(self.device)
        elif config["network_type"] == "conv1D":
            self.regressor = regressor_circular.CirConvNet().to(self.device)
        
        print(self.regressor)
        pytorch_total_params = sum(p.numel() for p in self.regressor.parameters() if p.requires_grad)
        print(f"Model Params: {pytorch_total_params}")
        self.optimizer = optim.Adam(self.regressor.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)
        self.epochs = config["epochs"]
        self.save_path = config["save_path"]
        self.best_model = None
    
    def accuracy(self, loader, model=None):
        if model is None:
            model = self.regressor
        acc_rmse = 0.0
        acc_mae = 0.0
        with torch.no_grad():
            for inp, labels in loader:
                inp = inp.to(self.device)
                labels = labels.to(self.device)
                pred = model(inp.float())
                loss = self.loss_func(pred[:, 0], labels.float())
                l1_loss = F.l1_loss(pred[:, 0], labels.float())
                acc_rmse += loss.item()
                acc_mae += l1_loss.item()
        return np.sqrt(acc_rmse/len(loader)), acc_mae/len(loader)
    
    def train(self):
        min_rmse = 1000000
        min_mae = 1000000
        best_val_epoch = -1
        best_val_mae = -1
        for epoch in range(self.epochs):
            train_loss = 0.0
            for inp, label in self.train_loader:
                inp = inp.to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                pred = self.regressor(inp.float())
                loss = F.mse_loss(pred[:, 0], label.float())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inp.shape[0]
            self.scheduler.step()
            with torch.no_grad():
                val_loss = 0.0
                val_accuracy_rmse, val_accuracy_mae = self.accuracy(self.val_loader) 
                if val_accuracy_rmse < min_rmse:
                    min_rmse = val_accuracy_rmse
                    self.best_model = copy.deepcopy(self.regressor)
                    best_val_epoch = epoch
                if val_accuracy_mae < min_mae:
                    min_mae = val_accuracy_mae
                    
                if not epoch % 100:
                    train_rmse, train_mae = self.accuracy(self.train_loader)
                    print(f"Epoch - {epoch}, train loss - {train_loss/len(self.train_set)*1.0:.5f}, Train RMSE - {train_rmse:.5f}, Train MAE - {train_mae:.5f}, Val RMSE - {val_accuracy_rmse:.5f}, Val MAE - {val_accuracy_mae:.5f}")
        if self.save_model:
            with torch.no_grad():
                model_info_path = os.path.join(self.save_path, str(today))
                print(f"Saving Model to {model_info_path}")
                if os.path.exists(model_info_path):
                    shutil.rmtree(model_info_path)
                os.makedirs(model_info_path)
                torch.save(self.best_model.cpu(), os.path.join(model_info_path, f"regressor_{today}.pt"))
                shutil.copyfile(self.config_path, os.path.join(model_info_path, "config.json"))
        print(f"Best Epoch: {best_val_epoch}, RMSE: {min_rmse} MAE: {min_mae}")
        test_rmse, test_mae = self.accuracy(self.test_loader, self.best_model)
        print(f"Test Accuracy (Best val acc model): rmse - {test_rmse}, mae - {test_mae}")
        
    def inference_on_csv(self, csv_path, model_path, upsampler_path=None):
        model = torch.load(model_path)
        if csv_path is None:
            print("Error: CSV path not given. Exiting")
            exit(0)
        input_signals = pd.read_csv(csv_path, header=None).transpose().to_numpy()
        if upsampler_path:
            upsampler = torch.load(upsampler_path)  
            upsampler.eval()  
            input_signals = upsampler(torch.from_numpy(input_signals).float()).detach().numpy()
        output_signals = model(torch.from_numpy(input_signals).float()).detach().numpy()
        output_signals_df = pd.DataFrame(output_signals)
        output_signals_df.to_csv("data/hrv_inferenced_reconstructed_1.csv", index=False, header=None)
    
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='HRV Regressor Training')
    parser.add_argument('--config', help="path to config with training params", default="config.json")
    parser.add_argument('--signal_type', help="Signal type", default="original", choices=["original", "reconstructed"])
    args = parser.parse_args()
    trainer =  HrvRegressor(config_path=args.config, signal_type=args.signal_type)
    trainer.train()
    # trainer.inference_on_csv('/Users/ganesh/UofA/SNN/freq_data_final_corrected/6Hz_normal.csv', '/Users/ganesh/UofA/SNN/HRV-edgedevice/models/regressor/2022-07-10-23:19:40/regressor_2022-07-10-23:19:40.pt', 'models/upsampler/2022-07-10-21:40:37/upsampler_2022-07-10-21:40:37.pt')