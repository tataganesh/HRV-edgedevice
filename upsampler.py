import torch
from data_utils import read_freq_data, SignalDataset
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import json
import argparse
import os
from datetime import datetime
from networks.net import NeuralNetwork
today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')



class AnnUpsampler:
    def __init__(self, config=None):
        torch.manual_seed(1)
        input_signal, output_signal, self.labels = read_freq_data(config["folder_path"], config["signal_percentage"])
        full_dataset = SignalDataset(input_signal, output_signal, self.labels, transforms.ToTensor())
        self.train_set, self.val_set, self.test_set = random_split(full_dataset, [650, 194, 195])
        # torch.save(np.array(self.train_set.indices), "data/train_set_indices.npy")
        # torch.save(np.array(self.val_set.indices), "data/val_set.npy")
        # torch.save(np.array(self.test_set.indices), "data/test_set.npy")
        # Data Loaders
        self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size = config["batch_size"])
        self.val_loader = DataLoader(self.val_set, shuffle=True, batch_size = config["batch_size"])
        self.test_loader = DataLoader(self.test_set, batch_size = len(self.test_set))

        # Loss Functiin
        self.loss_func = torch.nn.MSELoss()
        self.ann_upsampler = NeuralNetwork(input_signal.shape[1], output_signal.shape[1], config["layer_sizes"])
        print(self.ann_upsampler)
        self.optimizer = optim.SGD(self.ann_upsampler.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.epochs = config["epochs"]
        self.save_path = config["save_path"]

    def accuracy(self, loader):
        acc = 0.0
        with torch.no_grad():
            for inp, op, labels in loader:
                pred = self.ann_upsampler(inp.float())
                loss = self.loss_func(pred, op.float())
                acc += loss.item()
        return acc/len(loader)
        

    def train(self):
        for epoch in range(0, self.epochs):
            train_loss = 0.0
            for inp, op, label in self.train_loader:
                self.optimizer.zero_grad()
                pred = self.ann_upsampler(inp.float())
                loss = self.loss_func(pred, op.float())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            val_loss = 0.0

            if not epoch % 100:
                print(f"Epoch - {epoch}, train loss - {train_loss/len(self.train_loader)*1.0:.5f}, Train accuracy - {self.accuracy(self.train_loader)*1.0:.5f}, val accuracy - {self.accuracy(self.val_loader)*1.0:.5f}")
            
        with torch.no_grad():
            torch.save(self.ann_upsampler, os.path.join(self.save_path, f"upsampler_{today}.pt"))
        
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
            
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ANN Upsampler Training')
    parser.add_argument('--config', help="path to config with training params", required=True)
    args = parser.parse_args()
    config=json.load(open(args.config, 'r'))
    trainer =  AnnUpsampler(config=config)
    trainer.train()
    trainer.test()
    # trainer.upsample_and_save(torch.load("models/upsampler_2022-04-05-00:03:17.pt"))