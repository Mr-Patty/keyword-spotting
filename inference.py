import numpy as np
import pyaudio
import wave
import time
import onnxruntime
import argparse
import torch
import torchaudio
from utils import to_numpy

from torch import nn

CHUNK = 16000
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

# def get_prediction()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_set', default=['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'],
                        help='Set of labels')
    parser.add_argument('--checkpoint', default='models/key-spotting.onnx', help='Path to saved checkpoint')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--type', default='onnx', help='type of model onnx or torch')
    namespace = parser.parse_args()
    argv = vars(namespace)

    providers = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(argv['checkpoint'], providers=providers)
    audio_transforms = nn.Sequential(
        torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={'hop_length': 16 * 10,
                                                                            "center": True, 'n_mels': 40}),
        torchaudio.transforms.SlidingWindowCmn(cmn_window=600, norm_vars=True, center=True)
    )
    labels_set = ['silent', 'unk'] + argv['labels_set']

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
            torch_audio = torch.frombuffer(data, dtype=torch.float32)
            transform_audio = audio_transforms(torch_audio).reshape(-1, 40)
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch.unsqueeze(transform_audio, 0))}
            ort_outs = ort_session.run(None, ort_inputs)
            pred = ort_outs[0]
            j = pred.argmax(1)[0]
            print(labels_set[j])

            frames.append(data)
    except KeyboardInterrupt:

        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()