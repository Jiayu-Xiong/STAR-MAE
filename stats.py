import torch
import numpy as np
from datasets.ESC50 import get_ESC50_dataloader
from datasets.AudioSet_FT import get_dataset_20K
from datasets.SpeechCommandCustom import create_speechcommands_dataloaders
from datasets.Opera import get_opera_dataloader

audio_conf_esc50 = {
    'sample_rate': 16000,
    'n_fft': 2048,
    'H': 256,
    'W': 256,
    'T': 2,
    'overlap_rate': 0.0,
    'per_frame_overlap': False,
    'use_mel': True,
    'hop_length': None,
    'mixup': 0.5,
    'skip_norm': False,
    'noise': False,
    'patch_h': 16,
    'patch_w': 16,
    'tubelet': 1,
    'batchsize': 100,
    'workers': 24,
    'root': 'path',
    'encoder_msk_rate': 0.75, # discard
    'num_classes': 50
}

# train_loader, _ =  get_ESC50_dataloader(audio_conf_esc50, "0")
audio_conf_audioset = {
    'sample_rate': 16000,
    'n_fft': 2048,
    'H': 256,
    'W': 256,
    'T': 4,
    'overlap_rate': 0.0,
    'per_frame_overlap': False,
    'use_mel': True,
    'hop_length': 160,
    'mixup': 0.5,
    'skip_norm': False,
    'noise': False,
    'patch_h': 16,
    'patch_w': 16,
    'tubelet': 1,
    'batchsize': 8,
    'workers': 16,
    'root': 'path',
    'encoder_msk_rate': 0.75, # discard
    'decoder_msk_rate': 0.5,
    'encoder_msk_type': 'tube',
    'decoder_msk_type': 'random',
    'num_classes': 527
}

# train_loader, test_loader = get_dataset_20K(audio_conf=audio_conf_audioset)

audio_conf_scfull = {
    'sample_rate': 16000,
    'n_fft': 2048,
    'H': 256,
    'W': 128,
    'T': 1,
    'overlap_rate': 0.0,
    'per_frame_overlap': False,
    'use_mel': True,
    'hop_length': 160,
    'mixup': 0.5,
    'skip_norm': False,
    'noise': False,
    'patch_h': 16,
    'patch_w': 16,
    'tubelet': 1,
    'batchsize': 32,
    'workers': 24,
    'root': 'path',
    'encoder_msk_rate': 0.75, # discard
    'num_classes': 35
}

audio_conf_opera = {
    'sample_rate': 16000,
    'n_fft': 2048,
    'H': 256,
    'W': 256,
    'T': 2,
    'overlap_rate': 0.0,
    'per_frame_overlap': False,
    'use_mel': True,
    'hop_length': 160,
    'mixup': 0.5,
    'skip_norm': False,
    'noise': False,
    'patch_h': 16,
    'patch_w': 16,
    'tubelet': 1,
    'batchsize': 32,
    'workers': 24,
    'root': 'path',
    'encoder_msk_rate': 0.75, # discard
    'num_classes': 35
}
# Some parameters are introduced only to meet the format of the program
train_loader, _ = get_opera_dataloader(audio_conf_opera, test_fold=99)
mean=[]
std=[]

for i, (audio_input, labels) in enumerate(train_loader):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))
