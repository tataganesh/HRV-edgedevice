import torch
from data_utils import read_freq_data, SignalDataset

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
from net import NeuralNetwork
today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')



class AnnUpsampler:
    def __init__(self, config=None):
        torch.manual_seed(1)
        input_signal, output_signal, self.labels = read_freq_data(config["folder_path"])
        full_dataset = SignalDataset(input_signal, output_signal, self.labels, transforms.ToTensor())
        train_set, val_set, test_set = random_split(full_dataset, [650, 194, 195])
        print(train_set[0])
        exit()

        # Data Loaders
        self.train_loader = DataLoader(train_set, shuffle=True, batch_size = config["batch_size"])
        self.val_loader = DataLoader(val_set, shuffle=True, batch_size = config["batch_size"])
        self.test_loader = DataLoader(test_set, batch_size = len(test_set))

        # Loss Functiin
        self.loss_func = torch.nn.MSELoss()
        self.ann_upsampler = NeuralNetwork(input_signal.shape[1], output_signal.shape[1])
        self.optimizer = optim.SGD(self.ann_upsampler.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.epochs = config["epochs"]
        self.save_path = config["save_path"]

    def train(self):
        for epoch in range(0, self.epochs):
            train_loss = 0.0
            for inp, op, labels in self.train_loader:
                self.optimizer.zero_grad()
                pred = self.ann_upsampler(inp.float())
                loss = self.loss_func(pred, op.float())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            val_loss = 0.0
            with torch.no_grad():
                for inp, op, labels in self.val_loader:
                    pred = self.ann_upsampler(inp.float())
                    loss = self.loss_func(pred, op.float())
                    val_loss += loss.item()
            if not epoch % 100:
                print(f"Epoch - {epoch}, train loss - {train_loss/len(self.train_loader)*1.0:.5f}, val loss - {val_loss/len(self.val_loader)*1.0:.5f}")


        torch.save(self.ann_upsampler, os.path.join(self.save_path, f"upsampler_{today}.pt"))
        
    def test(self):
        with torch.no_grad():
            for test_inp, test_op, labels in self.test_loader:
                pred = self.ann_upsampler(test_inp.float())
                mse = self.loss_func(pred, test_op.float())
            print(f"Test MSE - {mse}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ANN Upsampler Training')
    parser.add_argument('--config', help="path to config with training params", required=True)
    args = parser.parse_args()
    config=json.load(open(args.config, 'r'))
    trainer =  AnnUpsampler(config=config)
    trainer.train()
    trainer.test()