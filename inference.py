import pyaudio
import onnxruntime
import argparse
import torch
import torchaudio
from utils import to_numpy
from model import SpeechResModel

from torch import nn

CHUNK = 16000
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5


class Inference:
    def __init__(self, checkpoint, n_labels, model_type='onnx', dilation=True, device='cpu', n_mels=64):
        providers = ['CPUExecutionProvider']
        self.type = model_type
        device = torch.device(device)
        if model_type == 'torch':
            self.model = SpeechResModel(n_labels=n_labels, dilation=dilation).float()
            self.model.load_state_dict(torch.load(checkpoint, map_location=device))
            self.model.eval()
        elif model_type == 'onnx':
            self.model = onnxruntime.InferenceSession(checkpoint, providers=providers)
        else:
            print('model type must be onnx or torch')
        self.audio_transforms = nn.Sequential(
            torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={'hop_length': 16 * 10,
                                                                                "center": True, 'n_mels': n_mels}),
            torchaudio.transforms.SlidingWindowCmn(cmn_window=600, norm_vars=True, center=True)
        )

    def get_prediction(self, audio):
        torch_audio = torch.frombuffer(audio, dtype=torch.float32)
        transform_audio = self.audio_transforms(torch_audio).reshape(1, -1, 40)
        if self.type == 'onnx':
            ort_inputs = {self.model.get_inputs()[0].name: to_numpy(transform_audio)}
            ort_outs = self.model.run(None, ort_inputs)
            pred = ort_outs[0]
            res = pred.argmax(1)[0]
        elif self.type == 'torch':
            pred = self.model(transform_audio)
            res = pred.argmax(1)[0].tolist()
        else:
            res = 0
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_set', default=['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'],
                        help='Set of labels')
    parser.add_argument('--checkpoint', default='models/model_v4.onnx', help='Path to saved checkpoint')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--type', default='onnx', help='type of model onnx or torch')
    parser.add_argument('--n_mels', default='64', help='number of mels')
    namespace = parser.parse_args()
    argv = vars(namespace)

    n_mels = int(argv['n_mels'])
    labels_set = ['silent', 'unk'] + argv['labels_set']

    inference = Inference(argv['checkpoint'], len(labels_set), model_type=argv['type'], device=argv['device'])

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    frames = []
    try:
        while True:
            data = stream.read(CHUNK)
            pred = inference.get_prediction(data)
            if 1 < pred < len(labels_set):
                print(labels_set[pred])
            frames.append(data)
    except KeyboardInterrupt:
        print("* done recording")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
