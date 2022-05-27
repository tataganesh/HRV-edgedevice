import glob
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch



def read_freq_data(folder_path, signal_percentage=1):
    sixhzsignal = pd.DataFrame({})
    twelvehzsignal = pd.DataFrame({})
    labels = list()

    # folder_path = "/content/drive/MyDrive/SNN_project/ANN Upsampling Project Data"
    for i in range(1, 4):
        twelvehz_normal = pd.read_csv(os.path.join(folder_path, f"6hznormal{i}.csv"), header=None).transpose()
        sixhz_normal = pd.read_csv(os.path.join(folder_path, f"12hznormal{i}.csv"), header=None).transpose()
        twelvehz_abnormal = pd.read_csv(os.path.join(folder_path, f"6hzabnormal{i}.csv"), header=None).transpose()
        sixhz_abnormal = pd.read_csv(os.path.join(folder_path, f"12hzabnormal{i}.csv"), header=None).transpose()
        print(twelvehz_normal.shape, sixhz_normal.shape, twelvehz_abnormal.shape, sixhz_abnormal.shape)
        if twelvehz_normal.shape[0] < sixhz_normal.shape[0]:
            sixhz_normal = sixhz_normal.iloc[0: twelvehz_normal.shape[0]]
        else:
            twelvehz_normal = twelvehz_normal.iloc[0: sixhz_normal.shape[0]]

        if twelvehz_abnormal.shape[0] < sixhz_abnormal.shape[0]:
            sixhz_abnormal = sixhz_abnormal.iloc[0: twelvehz_abnormal.shape[0]]
        else:
            twelvehz_abnormal = twelvehz_abnormal.iloc[0: sixhz_abnormal.shape[0]]
        if i == 1:
            sixhz_abnormal = sixhz_abnormal.iloc[0:0]
            twelvehz_abnormal = twelvehz_abnormal.iloc[0:0]
        sixhzsignal = pd.concat([sixhzsignal, sixhz_normal, sixhz_abnormal])
        twelvehzsignal = pd.concat([twelvehzsignal, twelvehz_normal, twelvehz_abnormal])
        labels.extend([0] * sixhz_normal.shape[0])
        labels.extend([1] * sixhz_abnormal.shape[0])

    input_signal = sixhzsignal.to_numpy()
    input_signal = input_signal[:, :int(input_signal.shape[1] * signal_percentage)]
    output_signal = twelvehzsignal.to_numpy()
    output_signal = output_signal[:, :int(output_signal.shape[1] * signal_percentage)]
    labels = np.array(labels)
    print(f"Normal Signals - {np.sum(labels==0)}")
    print(f"Abnormal Signals - {np.sum(labels==1)}")

    print(f"6Hz Signals shape - {input_signal.shape}")
    print(f"12Hz Signals shape - {output_signal.shape}")
    print(f"Labels - {labels.shape}")
    return input_signal, output_signal, labels

class SignalDataset(Dataset):
    def __init__(self, input_signal, output_signal, labels, transform):
        self.input_signal = input_signal
        self.output_signal = output_signal
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        inp_sig = torch.from_numpy(self.input_signal[index])
        op_sig = torch.from_numpy(self.output_signal[index])
        label = self.labels[index]
        return inp_sig, op_sig, label

    def __len__(self):
        return self.input_signal.shape[0]



if __name__ =="__main__":
    read_freq_data('/Users/ganesh/UofA/SNN/freq_data')