import os
#
from os import listdir
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from datetime import datetime

import IPython
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import collections
import contextlib
import sys
import wave
import json
import pickle
import random
# import torchaudio

import matplotlib.pyplot as plt
import librosa
#
import argparse
import torch
import json
import pickle

import torch.utils.data as data_utils
import soundfile as sf

from models.vad_models import LSTMModel
from datetime import datetime
from utils import *
from os import listdir
from tqdm import tqdm
from model import SpeechResModel

# base_dir = "dataset/"
# test_file_path = "testing_list.txt"
# validation_file_path = "validation_list.txt"
# noise_path = "_background_noise_/"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', default='dataset/', help='Path to dataset')
    parser.add_argument('--noise_name', default='_background_noise_/', help='Name of noise data')
    parser.add_argument('--valid_file_name', default='validation_list.txt', help='Name of validation file')
    parser.add_argument('--labels_set', default=['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'],
                        help='Set of labels')
    parser.add_argument('--checkpoints', default='checkpoints', help='Path to saved checkpoints')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--batch_size', default='128', help='batch size')
    parser.add_argument('--epochs', default='40', help='Number of train epochs')
    namespace = parser.parse_args()
    argv = vars(namespace)

    all_samples = []
    base_dir = argv['path_dataset']
    validation_file_path = os.path.join(base_dir, argv['valid_file_name'])
    noise_path = os.path.join(base_dir, argv['noise_name'])
    checkpoints_path = argv['checkpoints']
    device = argv['device']
    batch_size = int(argv['batch_size'])
    epochs = int(argv['epochs'])

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # labels_set = ['bed','bird','cat','dog','down','eight','five','four','go','happy','house',
    #               'left','marvin','nine','no','off','on','one','right','seven','sheila','six',
    #               'stop','three','tree','two','up','wow','yes','zero'
    #              ]
    # labels_set = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    labels_set = argv['labels_set']

    for word in listdir(base_dir):
        if os.path.isdir(word) and word in labels_set:
            for path in listdir(base_dir + word):
                all_samples.append(os.path.join(word, path))

    with open(validation_file_path) as file:
        list_samples = file.read()
    validation_samples = list_samples.split('\n')[:-1]

    train_samples = list(set(all_samples) - set(validation_samples) - set(test_samples))

    model = SpeechResModel(n_labels=len(labels_set) + 2).float()
    print("Start train model")
    trainModel(model, train_samples, validation_samples, checkpoints_path, noise_path, labels_set, base_dir,
               device=device, batch_size=batch_size, EPOCHS=epochs)
