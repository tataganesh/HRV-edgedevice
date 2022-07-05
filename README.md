# HRV-edgedevice
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zfv7Uj2l-QWJ9dpxFHZ4ikUdu0U3v5aT?usp=sharing)

DNN-based system to calculate HRV using low frequency input signal


## Running scripts

Upsampler

```python3 upsampler.py --config=config.json```

Classifier

```python3 classifier.py --folder_path /Users/ganesh/UofA/SNN/freq_data --upsampler /Users/ganesh/UofA/SNN/HRV-edgedevice/models/upsampler_2022-03-29-18:31:55.pt ```
