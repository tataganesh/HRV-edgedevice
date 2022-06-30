import glob
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms



def read_freq_data(folder_path, signal_percentage=1, include_abnormal=True):
    sixhzsignals = list()
    twelvehzsignals = list()
    labels = list()
    sixhz_normal = pd.read_excel(os.path.join(folder_path, f"6Hz_normal.xlsx"), header=None).transpose()
    twelvehz_normal = pd.read_excel(os.path.join(folder_path, f"12Hz_normal.xlsx"), header=None).transpose()
    print(f"Normal Signal  shape: {sixhz_normal.shape}, {twelvehz_normal.shape}")
    sixhzsignals.append(sixhz_normal)
    twelvehzsignals.append(twelvehz_normal)
    labels.extend([0] * sixhz_normal.shape[0])
    if include_abnormal:
        sixhz_abnormal = pd.read_excel(os.path.join(folder_path, f"6Hz_abnormal.xlsx"), header=None).transpose()
        twelvehz_abnormal = pd.read_excel(os.path.join(folder_path, f"12Hz_abnormal.xlsx"), header=None).transpose()
        print(f"Abnormal Signal shape: {sixhz_abnormal.shape}, {twelvehz_abnormal.shape}")
        sixhzsignals.append(sixhz_abnormal)
        twelvehzsignals.append(twelvehz_abnormal)
        labels.extend([1] * sixhz_abnormal.shape[0])
    labels = np.array(labels)
    sixhzsignal = pd.concat(sixhzsignals)
    twelvehzsignal = pd.concat(twelvehzsignals)

    input_signal = sixhzsignal.to_numpy()
    input_signal = input_signal[:, :int(input_signal.shape[1] * signal_percentage)]
    output_signal = twelvehzsignal.to_numpy()
    output_signal = output_signal[:, :int(output_signal.shape[1] * signal_percentage)]
    print(f"Normal Signals count- {np.sum(labels==0)}")
    print(f"Abnormal Signals count- {np.sum(labels==1)}")
    print(f"6Hz Signals shape - {input_signal.shape}")
    print(f"12Hz Signals shape - {output_signal.shape}")
    print(f"Labels - {labels.shape}")
    return input_signal.astype(np.float32), output_signal.astype(np.float32), labels

class SignalDataset(Dataset):
    def __init__(self, input_signal, output_signal, labels, transform=None, hrv_values=None):
        self.input_signal = input_signal
        self.output_signal = output_signal
        self.labels = labels
        self.hrv_values = hrv_values

    def __getitem__(self, index):
        inp_sig = torch.from_numpy(self.input_signal[index])
        op_sig = torch.from_numpy(self.output_signal[index])
        label = self.labels[index]
        if self.hrv_values is not None:
            hrv = self.hrv_values[index]
            return inp_sig, op_sig, label, hrv
        return inp_sig, op_sig, label

    def __len__(self):
        return self.input_signal.shape[0]


def get_all_sets(input_signal, output_signal, labels, hrv_data=None):
    full_dataset = SignalDataset(input_signal, output_signal, labels, transforms.ToTensor(), hrv_values=hrv_data)
    return random_split(full_dataset, [573, 192, 192])
    # return random_split(full_dataset, [521, 200, 150])


if __name__ =="__main__":
    read_freq_data('/Users/ganesh/UofA/SNN/freq_data')