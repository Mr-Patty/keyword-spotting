import argparse
import torch
import pickle
import onnx
import onnxruntime

import numpy as np
import torch.utils.data as data_utils

from sklearn.metrics import classification_report
from models.vad_models import LSTMModel
from tqdm import tqdm
from utils import *
from model import SpeechResModel
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', default='dataset/', help='Path to dataset')
    parser.add_argument('--noise_name', default='_background_noise_/', help='Name of noise data')
    parser.add_argument('--test_file_name', default='testing_list.txt', help='Name of test file')
    parser.add_argument('--labels_set', default=['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'],
                        help='Set of labels')
    parser.add_argument('--checkpoint', default='models/model_v4.pt', help='Path to saved checkpoint')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--type', default='torch', help='type of model onnx or torch')
    parser.add_argument('--batch_size', default='32', help='batch size')
    namespace = parser.parse_args()
    argv = vars(namespace)

    model_type = argv['type']
    base_dir = argv['path_dataset']
    test_file_path = os.path.join(base_dir, argv['test_file_name'])
    noise_path = os.path.join(base_dir, argv['noise_name'])
    checkpoints_path = argv['checkpoints']
    device = argv['device']
    batch_size = int(argv['batch_size'])
    labels_set = argv['labels_set']

    with open(test_file_path) as file:
        list_samples = file.read()
    test_samples = list_samples.split('\n')[:-1]

    if model_type == 'torch':
        model = SpeechResModel(n_labels=len(labels_set) + 2, dilation=True).float()
        model.load_state_dict(torch.load(argv['checkpoint']))
        model.eval()
    elif model_type == 'onnx':
        onnx_model = onnx.load(argv['checkpoint'])
        onnx.checker.check_model(onnx_model)
        model = onnxruntime.InferenceSession(argv['checkpoint'])
    else:
        print('model type must be onnx or torch')

    test_y, preds = testModel(model, test_samples, noise_path, labels_set, base_dir, device=device,
                              batch_size=batch_size, n_mels=64, model_type=model_type)
    test_y = np.hstack(test_y)
    preds = torch.softmax(torch.from_numpy(np.vstack(preds)), dim=1).numpy()
    acc = accuracy_score(test_y, preds.argmax(1))
    print("Accuracy: {:.2f}%".format(acc*100))

    print("Classification report:")
    print(classification_report(test_y, preds.argmax(1).astype(int)))
