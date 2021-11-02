import os

from os import listdir
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from datetime import datetime

import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from audiomentations import Compose, TimeStretch, PitchShift, Shift
from sklearn.metrics import accuracy_score

import random
import torchaudio


class SpeechDataset(data.Dataset):
    def __init__(self, data, set_type, noise_path, labels_set, base_path, unknown_prob=0.1, silence_prob=0.1,
                 noise_prob=0.8, timeshift_ms=100, input_length=16000, n_mels=40, n_mfcc=40, hop_ms=10):
        super().__init__()
        LABEL_SILENCE = "__silence__"
        LABEL_UNKNOWN = "__unknown__"
        self.base_path = base_path
        self.noise_path = noise_path
        self.audio_files = data
        self.set_type = set_type
        labels = list(map(lambda x: x[:x.find('/')], data))
        self.label2ind = {word: i + 2 for i, word in enumerate(labels_set)}
        self.label2ind.update({LABEL_SILENCE: 0, LABEL_UNKNOWN: 1})
        self.audio_labels = list(map(lambda x: self.label2ind.get(x, 1), labels))
        self.n_mfcc = n_mfcc

        bg_noise_files = list(filter(lambda x: x.endswith("wav"), listdir(noise_path)))
        self.bg_noise_audio = [sf.read(os.path.join(noise_path, file))[0] for file in bg_noise_files]
        self.unknown_prob = unknown_prob
        self.silence_prob = silence_prob
        self.noise_prob = noise_prob
        self.input_length = input_length
        self.timeshift_ms = timeshift_ms
        self._file_cache = {}
        self._audio_cache = {}
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))

        self.augment = Compose([
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_fraction=-0.1, max_fraction=0.1, p=0.5, rollover=False),
        ])

        self.audio_transforms = nn.Sequential(
            torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mfcc, melkwargs={'hop_length': 16 * hop_ms,
                                                                                    "center": True, 'n_mels': n_mels}),
            torchaudio.transforms.SlidingWindowCmn(cmn_window=600, norm_vars=True, center=True)
        )
        self.train_audio_augmentations = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=3),
            torchaudio.transforms.TimeMasking(time_mask_param=10)
        )

    def load_audio(self, example, silence=False):
        if silence:
            example = "__silence__"
        if random.random() < 0.7 or not self.set_type == 'train':
            try:
                return self._audio_cache[example]
            except KeyError:
                pass
        in_len = self.input_length
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a:a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        if silence:
            audio = np.zeros(in_len, dtype=np.float32)
        else:
            file_data = self._file_cache.get(example)
            audio = sf.read(os.path.join(self.base_path, example))[0] if file_data is None else file_data
            audio = audio.astype(np.float32)
            self._file_cache[example] = audio
        audio = np.pad(audio, (0, max(0, in_len - len(audio))), "constant")
        if self.set_type == 'train':
            audio = self.augment(samples=audio, sample_rate=16000)

        if random.random() < self.noise_prob or silence:
            if silence:
                a = random.random() * 0.4
            else:
                a = random.random() * 0.1
            audio = np.clip(a * bg_noise + audio, -1, 1)

        torch_audio = torch.from_numpy(audio).float()
        transform_audio = self.audio_transforms(torch_audio).reshape(-1, self.n_mfcc)
        if self.set_type == 'train':
            transform_audio = self.train_audio_augmentations(transform_audio)
        self._audio_cache[example] = transform_audio
        return transform_audio

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.load_audio(None, silence=True), torch.tensor(0)
        return self.load_audio(self.audio_files[index]), torch.tensor(self.audio_labels[index])

    def __len__(self):
        return len(self.audio_labels) + self.n_silence


def trainModel(model, train_samples, validation_samples, checkpoints_path, noise_path, labels_set, base_dir, lr=1e-3,
               EPOCHS=10, batch_size=64, device='cuda', each=10, logging=False):

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=2, threshold=0.01)
    last_epoch = -1

    train_dataset = SpeechDataset(train_samples, 'train', noise_path, labels_set, base_dir, n_mels=64)
    use_cuda = device == 'cuda'
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    print('DATASET SIZE: {}'.format(len(train_dataset)))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, drop_last=True, **kwargs)

    validation_dataset = SpeechDataset(validation_samples, 'test', noise_path, labels_set, base_dir, n_mels=64)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False, **kwargs)

    model.train()
    iterator = tqdm(range(EPOCHS), desc='epochs')
    print('START TRAINING')
    max_acc = 0

    for epoch in iterator:
        try:
            if epoch <= last_epoch:
                continue

            mean_loss = 0
            model.train()
            for batch_x, batch_y in tqdm(train_loader):
                optim.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.to(device)

                output = model(batch_x)
                loss = loss_function(output, batch_y)

                loss.backward()
                optim.step()

                mean_loss += loss.detach().cpu().item() * len(batch_x)
            mean_loss /= len(train_dataset)

            model.eval()
            preds = []
            test_y = []
            mean_loss_val = 0
            for batch_x, batch_y in tqdm(validation_loader):
                test_y.append(batch_y.numpy())
                with torch.no_grad():
                    output = model(batch_x.float().to(device))
                    loss = loss_function(output, batch_y.to(device))
                    pred = output.cpu().detach().numpy()
                    preds.extend(pred)

                    mean_loss_val += loss.detach().cpu().item() * len(batch_x)
            mean_loss_val /= len(validation_dataset)

            test_y = np.hstack(test_y)
            preds = np.vstack(preds)
            acc = accuracy_score(test_y, preds.argmax(1))
            scheduler.step(mean_loss_val)
            if epoch != 0 and epoch % each == 0 or (acc > max_acc and epoch > 10):
                max_acc = max(acc, max_acc)

                check_path = os.path.join(checkpoints_path, 'model_checpoint{}'.format(
                    datetime.now().strftime("_%Y%m%d_%H%M%S")) + '_{}'.format(epoch) + '.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': mean_loss,
                }, check_path)
            iterator.set_postfix({'train_loss': mean_loss, 'valid_loss': mean_loss_val, 'valid_acc': acc})
            if logging:
                with open('logging.txt', 'a+') as file:
                    file.write('{} {} {} {}\n'.format(mean_loss, mean_loss_val, acc, epoch))
        except KeyboardInterrupt:
            PATH = os.path.join(checkpoints_path,
                                'model_checpoint{}'.format(datetime.now().strftime("_%Y%m%d_%H%M%S")) + '_{}'.format(
                                    epoch) + '.pt')
            torch.save(model.state_dict(), PATH)
            return


def testModel(model, test_samples, noise_path, labels_set, base_dir, max_samples=10000, batch_size=1, device='cpu',
              n_mels=40, model_type='torch'):


    test_dataset = SpeechDataset(test_samples, 'test', noise_path, labels_set, base_dir, n_mels=n_mels)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    preds = []
    test_y = []
    print('DATASET SIZE: {}'.format(len(test_dataset)))
    if model_type == 'torch':
        model.to(device)
        model.eval()
        for x, y in tqdm(test_loader):
            with torch.no_grad():
                test_y.append(y.numpy())
                output = model(x.float().to(device))
                pred = output.cpu().detach().numpy()
                preds.extend(pred)
    elif model_type == 'onnx':
        for x, y in tqdm(test_loader):
            test_y.append(y.numpy())
            ort_inputs = {model.get_inputs()[0].name: to_numpy(x)}
            ort_outs = model.run(None, ort_inputs)
            pred = ort_outs[0]
            preds.extend(pred)
    del test_dataset
    return test_y, preds


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
