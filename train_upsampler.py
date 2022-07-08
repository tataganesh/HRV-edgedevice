import torch
from zmq import device
from data_utils import read_freq_data, get_all_sets
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import json
import argparse
import os
from datetime import datetime
from networks.net import NeuralNetwork
import shutil
today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')



class AnnUpsampler:
    def __init__(self, config_path=None):
        torch.manual_seed(1)
        self.config_path = config_path
        config=json.load(open(config_path, 'r'))
        self.save_model = config["save_model"]
        input_signal, output_signal, self.labels = read_freq_data(config["folder_path"])
        self.train_set, self.val_set, self.test_set = get_all_sets(input_signal, output_signal, self.labels)
        # Data Loaders
        self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size = config["batch_size"])
        self.val_loader = DataLoader(self.val_set, shuffle=True, batch_size = config["batch_size"])
        self.test_loader = DataLoader(self.test_set, batch_size = len(self.test_set))
                
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # Loss Function
        self.loss_func = torch.nn.MSELoss()
        self.ann_upsampler = NeuralNetwork(input_signal.shape[1], output_signal.shape[1], config["layer_sizes"]).to(self.device)
        print(self.ann_upsampler)
        self.optimizer = optim.SGD(self.ann_upsampler.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.epochs = config["epochs"]
        self.save_path = config["save_path"]



    def accuracy(self, loader):
        acc = 0.0
        with torch.no_grad():
            for inp, op, labels in loader:
                inp = inp.to(self.device)
                op = op.to(self.device)
                pred = self.ann_upsampler(inp.float())
                loss = self.loss_func(pred, op.float())
                acc += loss.item()
        return acc/len(loader)
        

    def train(self):
        for epoch in range(0, self.epochs):
            train_loss = 0.0
            for inp, op, label in self.train_loader:
                inp = inp.to(self.device)
                op = op.to(self.device)
                self.optimizer.zero_grad()
                pred = self.ann_upsampler(inp.float())
                loss = self.loss_func(pred, op.float())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            val_loss = 0.0

            if not epoch % 100:
                print(f"Epoch - {epoch}, train loss - {train_loss/len(self.train_loader)*1.0:.5f}, Train accuracy - {self.accuracy(self.train_loader)*1.0:.5f}, val accuracy - {self.accuracy(self.val_loader)*1.0:.5f}")
        self.test()
        if self.save_model:    
            with torch.no_grad():
                model_info_path = os.path.join(self.save_path, str(today))
                print(f"Saving Model to {model_info_path}")
                if os.path.exists(model_info_path):
                    shutil.rmtree(model_info_path)
                os.makedirs(model_info_path)
                torch.save(self.ann_upsampler.cpu(), os.path.join(model_info_path, f"upsampler_{today}.pt"))
                shutil.copyfile(self.config_path, os.path.join(model_info_path, "config.json"))
        
    def test(self):
        with torch.no_grad():
            for test_inp, test_op, labels in self.test_loader:
                pred = self.ann_upsampler(test_inp.float())
                mse = self.loss_func(pred, test_op.float())
            print(f"Test MSE - {mse}")
    
    def upsample_and_save(self, model=None):
        if model is None:
            model = self.ann_upsampler
        train_upsampled_signals = list()
        val_upsampled_signals = list()
        test_upsampled_signals = list()
        with torch.no_grad():
            train_upsampled_signals = np.array([model(inp.float()).numpy() for inp, _, _ in self.train_set])
            val_upsampled_signals =  np.array([model(inp.float()).numpy() for inp, _, _ in self.val_set])
            test_upsampled_signals =  np.array([model(inp.float()).numpy() for inp, _, _ in self.test_set])
            np.save("data/upsampled_signals_train.npy", train_upsampled_signals)
            np.save("data/upsampled_signals_val.npy", val_upsampled_signals)
            np.save("data/upsampled_signals_test.npy", test_upsampled_signals)
        for inp, op, _ in self.test_loader:
                mse = self.loss_func(torch.from_numpy(test_upsampled_signals), op.float())
                print(f"Test MSE - {mse}")
    
    def inference_on_csv(self, model=None, xlx_path=None):
        if model is None:
            model = self.ann_upsampler
        if xlx_path is None:
            print("Error: CSV path not given. Exiting")
            exit(0)
        input_signals = pd.read_excel(xlx_path, header=None).transpose().to_numpy()        
        output_signals = model(torch.from_numpy(input_signals).float()).detach().numpy()
        output_signals_df = pd.DataFrame(output_signals.T)
        # np.savetxt("12hznormal_reconstructed.csv", output_signals.detach().numpy().T, delimiter=",")
        output_signals_df.to_excel("12hznormal_reconstructed.xlsx", index=False, header=None)
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ANN Upsampler Training')
    parser.add_argument('--config', help="path to config with training params", required=True)
    args = parser.parse_args()
    trainer =  AnnUpsampler(config_path=args.config)
    trainer.train()
    # trainer.test()
    # trainer.upsample_and_save(torch.load("/Users/ganesh/UofA/SNN/HRV-edgedevice/models/upsampler/2022-06-12-19:54:42/upsampler_2022-06-12-19:54:42.pt"))
    # trainer.inference_on_csv(torch.load("/Users/ganesh/UofA/SNN/HRV-edgedevice/models/upsampler/2022-06-22-09:28:16/upsampler_2022-06-22-09:28:16.pt"), '/Users/ganesh/UofA/SNN/freq_data/6Hz_normal.xlsx')