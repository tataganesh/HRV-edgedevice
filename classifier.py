import numpy as np
from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import os
from data_utils import SignalDataset, read_freq_data
from torch.utils.data import random_split
import torchvision.transforms as transforms
import argparse
from networks.net import NeuralNetwork
import matplotlib.pyplot as plt
from joblib import dump
from datetime import datetime
from micromlgen import port
today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')


parser = argparse.ArgumentParser(description='ANN Upsampler Training')
parser.add_argument('--folder_path', help="Path to dataset folder", required=True)
parser.add_argument('--method', help="type of classifier to train", default="svm")
parser.add_argument('--upsampler', help="Path to upsampler model")
parser.add_argument('--save_path', help="Save Path for classifier", default="models")

args = parser.parse_args()
torch.manual_seed(1)
input_signal, output_signal, labels = read_freq_data(args.folder_path, 0.5) # The percentage argument needs to be provided via config
full_dataset = SignalDataset(input_signal, output_signal, labels, transforms.ToTensor())
train_set, val_set, test_set = random_split(full_dataset, [650, 194, 195])
ann_upsampler = torch.load(args.upsampler)
get_input = lambda x: np.array([ann_upsampler(inp.float()).detach().numpy() for inp, op, label in x])
get_label = lambda x: np.array([label for inp, op, label in x])

train_input = get_input(train_set)
train_labels = get_label(train_set)
test_input = get_input(test_set)
test_labels = get_label(test_set)
print(f"Train Normal Signals - {np.sum(train_labels==0)}")
print(f"Train Abnormal Signals - {np.sum(train_labels==1)}")

print(f"Test Normal Signals - {np.sum(test_labels==0)}")
print(f"Test Abnormal Signals - {np.sum(test_labels==1)}")

if args.method == "svm":
    clf = SVC(gamma=1.0/(train_input.shape[1] * train_input.var()), class_weight='balanced')
    print(1.0/(train_input.shape[1] * train_input.var()))
    clf.fit(train_input, train_labels)
elif args.method == "lr":
    clf = LogisticRegression(C=0.1, class_weight='balanced')
    # print(1.0/(train_input.shape[1] * train_input.var()))
    clf.fit(train_input, train_labels)
# exit()
# Save Model
# dump(clf, os.path.join(args.save_path, f'{args.method}_classifier_{today}.joblib'), compress=7)

# Save plain C version
with open(os.path.join(args.save_path, 'logisticregression_classifier.h'), 'w') as f:
    f.write(port(clf))
    
# Report test accuracy and display confusion matrix
print("Test Accuracy")
print((clf.predict(test_input) == test_labels).mean())
print(f1_score(test_labels, clf.predict(test_input)))
cm = confusion_matrix(test_labels, clf.predict(test_input))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Normal", 'Abnormal'])
disp.plot()


plt.show()