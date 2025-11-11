# get the pretrained model
# feed in the parkinson data (leave on patient out)
# use the adapter to upsample with interpolation
# pass the input through the trained encoder
# use the encoder weights for multi label classification
# finetune model on cross-entropy loss
# evaluate on the left out patient - f1 score and sleep scoring system (repeat this finetuning for all 12 patients later on and take the average of the left out patients)

import time
import pandas as pd
from pathlib import Path
import gc
import math
import matplotlib.pyplot as plt
from collections import namedtuple
from dataclasses import dataclass
from statistics import median
from typing import Union, Optional
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import subprocess
import transformers
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
import wandb
from transformer_encoder_decoder import Wav2Vec2ForSequenceClassification, _compute_mask_indices, WhisperModel, _compute_last_fixed_mask
from transformers import is_wandb_available, get_scheduler, Wav2Vec2FeatureExtractor

print(f"finished imports {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
#wandb.login(key="781ffa93f4f0316f1be9b044cb3302b5bfb87af7")


logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_train_steps", type=int, required=True)
    parser.add_argument("--num_warmup_steps", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--learning_rate_t", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)

    parser.add_argument("--has_decoder", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--loss_type", type=str, required=True)
    parser.add_argument("--mission", type=str, required=True)
    parser.add_argument("--data_length", type=int, required=True)

    parser.add_argument("--max_duration_in_seconds", type=float, required=True)
    parser.add_argument("--min_duration_in_seconds", type=float, required=True)
    parser.add_argument("--logging_steps", type=int, required=True)
    parser.add_argument("--saving_steps", type=int, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--per_device_test_batch_size", type=int, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True)
    parser.add_argument("--adam_beta1", type=float, required=True)
    parser.add_argument("--adam_beta2", type=float, required=True)
    parser.add_argument("--adam_epsilon", type=float, required=True)
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--mask_time_prob", type=float, required=True)
    parser.add_argument("--mask_time_length", type=int, required=True)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--validation_split_percentage", type=float, required=True)
    parser.add_argument("--lr_scheduler_type", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--push_to_hub", type=lambda x: x.lower() == "true", required=True)

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--train_label_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--test_label_path", type=str, required=True)

    parser.add_argument("--weights", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--encoder_sit",  type=str, required=True)
    parser.add_argument("--freeze_decoder", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--feature_ext", type=str, required=True)
    parser.add_argument("--model_pathway", type=str, required=True)
    parser.add_argument("--booster", type=float, required=True)

    parser.add_argument("--vers", type=str, required=True)
    parser.add_argument("--notes", type=str, required=True)

    return parser.parse_args()

print(f"finished parsing {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
args = parse_args()
print(args.data_length, flush=True)
tags = ['finetuning']

if args.encoder_sit == 'frozen_enc':
    tags.append('frozen encoder')
elif args.encoder_sit == 'cnn_only':
    tags.append('cnn_only')
elif args.encoder_sit == 'unmasked_enc':
    tags.append('unmasked_enc')

if args.feature_ext == 'cnn_frozen':
    tags.append('frozen CNN')
elif args.feature_ext == 'stft':
    tags.append('stft')
elif args.feature_ext == 'window_shifting':
    tags.append('window_shifting')

tags.append(args.loss_type)

if args.mission == 'next':
    tags.append('forecasting')

if args.loss_type == 'custom_ce':
    tags.append('custom_ce')

if args.has_decoder:
    tags.append('decoder + linear')

if 'pd' in args.model_pathway:
    tags.append('pd trained')
elif 'eeg' in args.model_pathway:
    tags.append('eeg-hdb trained')
elif 'edf' in args.model_pathway:
    tags.append('sleep-edf trained')
elif 'none' in args.model_pathway or 'r_d' in args.model_pathway:
    tags.append('no pretrained model')
    if 'r_d' in args.model_pathway:
        tags.append('downsampled')
if 'pd' in args.train_path and 'zscored' in args.train_path:
    tags.append('z scored pd data')
elif 'pd' in args.train_path and '0-1' in args.train_path:
    tags.append('0-1 norm pd data')
elif 'pd' in args.train_path and '2min' in args.train_path:
    tags.append('2min pd data')
elif 'ehb' in args.train_path:
    tags.append('eeg-hdb data')
elif 'sleep_edf' in args.train_path:
    tags.append('sleep-edf data')

if args.weights:
    tags.append(f'weights_with_b{args.booster}')

tags.append(f'lr_scheduler_{args.lr_scheduler_type}')

import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "/home/mengkris/links/scratch/wandb"
import time, random
time.sleep(random.uniform(1, 20))
run = wandb.init(
    project = "brain2vec_test",
    tags = tags,
    notes = args.notes,
    settings=wandb.Settings(init_timeout=300)
)

@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: Wav2Vec2ForSequenceClassification
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10
    has_cnn: Optional[bool] = True

    def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]
        if self.has_cnn:
            mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        else:
            batch["input_values"] = batch["input_values"]#.transpose(2,1)
            mask_indices_seq_length = batch["input_values"].shape[-2]
        
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None and self.has_cnn:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )
        elif not self.has_cnn:
            batch["sub_attention_mask"] = torch.ones(batch_size, mask_indices_seq_length, dtype=torch.bool, device=batch["input_values"].device)
            batch["attention_mask"] = torch.ones(batch_size, mask_indices_seq_length, dtype=torch.bool, device=batch["input_values"].device)
        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)

        return batch



train_path = args.train_path
train_label_path = args.train_label_path
test_path = args.test_path
test_label_path = args.test_label_path
if 'pd' in args.train_path:
    filename = Path(args.train_path).stem # get filename without extension
    patient_name = '_'.join(filename.split('_')[:2])
else:
    patient_name = 'subject'
trainy = np.load(train_path, allow_pickle=True)
trainy_label = np.load(train_label_path, allow_pickle=True)
# train_idx, val_idx = train_test_split(np.arange(len(trainy)), test_size=args.validation_split_percentage, random_state=42, shuffle=True, stratify=trainy_label)
#train = trainy[train_idx]
#train_label = trainy_label[train_idx]
#validation = trainy[val_idx]
#validation_label = trainy_label[val_idx]
test = np.load(test_path, allow_pickle=True)
test_label = np.load(test_label_path, allow_pickle=True)
# Downsample if model is edf and training data is pd
fs = 250
if ('r_d' in args.model_pathway or 'edf' in args.model_pathway) and 'pd' in args.train_path:
    from scipy.signal import resample
    fs = 100
    print("Downsampling train and validation sets to 3000 samples each (edf model, pd data)", flush=True)
    trainy = np.array([{'input_values': resample(i['input_values'], 3000, axis = -1)} for i in trainy])
    #validation = np.array([{'input_values': resample(i['input_values'], 3000, axis = -1)} for i in validation])
    test = np.array([{'input_values': resample(i['input_values'], 3000, axis = -1)} for i in test])

eeg_dataset ={}

from scipy.signal import stft, get_window
def stft_transform(dataset, fs, mode='no_band', window_size=2, overlap=1):
    """
    dataset: array of dicts with {'input_values': list of floats}
    fs: sampling rate
    window_size: seconds per STFT window
    overlap: seconds overlap
    Returns: array of shape (N, 7, time) — 7 frequency bands per time step
    """
    # Define the 7 bands (Hz)
    bands = [(0, 3), (3, 7), (7, 13), (13, 20), (20, 30), (30, 90), (90, 125)] # -> the frequency bands are inspired from joel's previos paper

    nperseg = int(window_size * fs)
    noverlap = int(overlap * fs)
    window = get_window("hamming", nperseg)
    stft_list = []
    for sample in dataset:
        raw_signal = np.array(sample['input_values'], dtype=np.float32)

        # Compute STFT
        f, t, Zxx = stft(
            raw_signal,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None
        )
        # Compute magnitude
        Zxx_abs = np.abs(Zxx)  # shape: (freq_bins, time_frames)

        if mode == 'band':
            # Aggregate into bands
            band_data = np.zeros((len(bands), Zxx_abs.shape[1]), dtype=np.float32)

            for i, (f_low, f_high) in enumerate(bands):
                idx = np.where((f >= f_low) & (f < f_high))[0]  # indices of bins in this band
                if len(idx) > 0:
                    band_data[i, :] = Zxx_abs[idx, :].mean(axis=0)
                else:
                    band_data[i, :] = 0  # if no bins fall in this band

            stft_list.append({'input_values':band_data})
        else:
            stft_list.append({'input_values': Zxx_abs.transpose()})


    return np.array(stft_list)  # shape: (N, 7, time_frames)
import numpy as np
from scipy.signal import firwin, filtfilt

def bandpass_fir_filter(eeg_signal, fs, bands, numtaps=501):
    """
    Bandpass filter EEG signal into multiple frequency bands using FIR filters.

    Args:
        eeg_signal (np.ndarray): 1D array of EEG signal.
        fs (int): Sampling frequency (Hz).
        bands (list of tuples): Frequency bands [(low1, high1), (low2, high2), ...].
        numtaps (int): Number of FIR filter taps (filter length, controls sharpness).

    Returns:
        filtered_dict (dict): Dictionary of filtered and normalized signals per band.
        cnn_input (np.ndarray): Stacked array of shape (num_bands, signal_length).
    """
    filtered_dict = {}
    for i, (low, high) in enumerate(bands):
        # Avoid zero cutoff (FIR requires >0 for highpass edge)
        low = max(low, 0.001)

        # Design FIR bandpass filter
        b = firwin(numtaps=numtaps, cutoff=[low, high], pass_zero=False, fs=fs)

        # Apply zero-phase filtering
        filtered = filtfilt(b, [1.0], eeg_signal)

        # Normalize each band (zero mean, unit variance)
        filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)

        filtered_dict[f'band_{i+1}_{low}-{high}Hz'] = filtered

    # Stack into shape (num_bands, signal_length)
    cnn_input = np.stack(list(filtered_dict.values()), axis=0)
    return filtered_dict, cnn_input


# Example usage:
#fs = 250  # sampling frequency in Hz
bands = [(0, 3), (3, 7), (7, 13), (13, 20), (20, 30), (30, 90), (90, 125)]

# Example EEG signal
# eeg_signal = np.load('eeg_sample.npy')  # or any 1D numpy array
# filtered_dict, cnn_input = bandpass_fir_filter(eeg_signal, fs, bands)

# cnn_input.shape -> (7, len(eeg_signal))

def custom_segments_with_shift(windows, window_size, shift, key):
    """
    windows: list of window arrays (each length = window_size)
    window_size: length of one window (e.g., 30 sec)
    shift: how much to shift labels (e.g., 15 sec)
    key: the key to access data in window dict, or None for labels
    """
    segments = []
    n = len(windows)
    
    # Calculate the shift ratio (0.5 for 15sec shift on 30sec windows)
    shift_ratio = shift / window_size
    
    # We need at least 2 windows to create shifted segments
    for i in range(n - 1):
        if key is None:
            # For labels: take second half of current window + first half of next window
            # This shifts the label forward by 'shift' seconds
            segments.append(windows[i])  # You might want to modify this based on your exact needs
        else:
            # For input: concatenate second half of window i with first half of window i+1
            first_window = windows[i][key]
            second_window = windows[i + 1][key]
            
            # Calculate split point
            split_point = int(len(first_window) * shift_ratio)
            
            # Take second half of first window + first half of second window
            seg = np.concatenate([
                first_window[split_point:],
                second_window[:split_point]
            ], axis=-1)
            
            segments.append({key: seg})
    
    return segments

def custom_segments(windows, target_len, mask_len, window_size, key):
    """
    windows: list of window arrays (each length = window_size)
    target_len: total segment length you want (e.g., 45 sec)
    window_size: length of one window (e.g., 30 sec)
    """

    full_needed = target_len // window_size
    remainder = target_len % window_size

    segments = []
    n = len(windows)

    for i in range(full_needed, n-1):
        ax = -1
        if args.feature_ext == 'stft':
            ax = -2
        if key is None: # so it's the label, and we want to take in the next label
            if remainder == 0:
                segments.append(windows[i])
            else:
                segments.append(windows[i+1])
        else:
            if remainder == 0:
                # only full windows
                seg = np.concatenate([windows[idx][key] for idx in range(i - full_needed , i)], axis=ax)
                if mask_len is not None:
                    seg = np.concatenate([seg, windows[i][key][:int(len(windows[i][key])*((mask_len)/window_size))]])
            else:
                # remainder + full windows
                seg = np.concatenate(
                    [windows[i-full_needed][key][int(len(windows[i][key])*((window_size-remainder)/window_size)):]]
                    + [windows[idx][key] for idx in range(i - full_needed + 1, i+1)], axis=ax
                )
                if mask_len is not None:
                    seg = np.concatenate([seg, windows[i+1][key][:int(len(windows[i][key])*((mask_len)/window_size))]])
            segments.append({key:seg})

    return segments



if args.feature_ext == 'stft': # or data length stuff
    trainy= stft_transform(trainy, fs)
    #validation, validation_label = stft_transform(validation), validation_label
    test = stft_transform(test, fs)

if args.mission =='next':
    trainy = np.array(custom_segments(trainy, args.data_length, None, window_size=30, key="input_values"))
    #validation = np.array(custom_segments(validation, args.data_length, None, window_size=30, key="input_values"))
    test = np.array(custom_segments(test, args.data_length , None, window_size=30, key="input_values"))
    trainy_label = np.array(custom_segments(trainy_label, args.data_length, None, window_size=30, key=None))
    #validation_label = np.array(custom_segments(validation_label, args.data_length, None, window_size=30, key=None))
    test_label = np.array(custom_segments(test_label, args.data_length, None, window_size=30, key=None))

if args.mission == 'shift':
    trainy = np.array(custom_segments_with_shift(trainy, window_size=30, shift=15, key="input_values"))
    #validation = np.array(custom_segments_with_shift(validation, window_size=30, shift=15, key="input_values"))
    test = np.array(custom_segments_with_shift(test, window_size=30, shift=15, key="input_values"))
    trainy_label = np.array(custom_segments_with_shift(trainy_label, window_size=30, shift=15, key=None))
    #validation_label = np.array(custom_segments_with_shift(validation_label, window_size=30, shift=15, key=None))
    test_label = np.array(custom_segments_with_shift(test_label, window_size=30, shift=15, key=None))

train_idx, val_idx = train_test_split(np.arange(len(trainy)), test_size=args.validation_split_percentage, random_state=42, shuffle=True, stratify=trainy_label)
train = trainy[train_idx]
train_label = trainy_label[train_idx]
validation = trainy[val_idx]
validation_label = trainy_label[val_idx]

print(f"Loading the data.... {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
eeg_dataset["train"] = np.array(train)
eeg_dataset["train_label"] = np.array(train_label)
eeg_dataset["validation"] = np.array(validation)
eeg_dataset["validation_label"] = np.array(validation_label)
eeg_dataset["test"] = np.array(test)
eeg_dataset["test_label"] = np.array(test_label)

if not args.weights:
    print(f"Balancing the dataset... {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    # Group by label
    class_indices = {label: np.where(train_label == label)[0] for label in range(5)}
    for label, indices in class_indices.items():
        print(f"Class {label} has {len(indices)} samples {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    class_samples = {label: train[indices] for label, indices in class_indices.items()}
    class_labels = {label: train_label[indices] for label, indices in class_indices.items()}
    balanced_train = []
    balanced_labels = []
    if 'down' in args.vers: # downsampling the data to the minimum class count
        min_count = min(len(indices) for indices in class_indices.values())
        # Replicate each class to match max_count

        for label in range(5):
            X = class_samples[label]
            y = class_labels[label]
            # Shuffle before selecting
            indices = np.random.permutation(len(X))[:min_count]
            balanced_train.append(X[indices])
            balanced_labels.append(y[indices])
        # Concatenate all classes back together
        eeg_dataset["train"] = np.concatenate(balanced_train)
        eeg_dataset["train_label"] = np.concatenate(balanced_labels)    
    else: # upsampling the data to the maximum class count
        # Get maximum class count
        max_count = max(len(indices) for indices in class_indices.values())


        for label in range(5):
            X = class_samples[label]
            y = class_labels[label]
            reps = max_count // len(X)
            remainder = max_count % len(X)
    
            # Repeat and pad
            replicated_X = np.concatenate([X] * reps + [X[:remainder]])
            replicated_y = np.concatenate([y] * reps + [y[:remainder]])
    
            balanced_train.append(replicated_X)
            balanced_labels.append(replicated_y)

        # Stack and shuffle
        train_balanced = np.concatenate(balanced_train)
        label_balanced = np.concatenate(balanced_labels)

        # Shuffle consistently
        indices = np.arange(len(train_balanced))
        np.random.shuffle(indices)

        eeg_dataset["train"] = train_balanced[indices]
        eeg_dataset["train_label"] = label_balanced[indices]
    print(f"Dataset balanced {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
else:  # if we apply weights to cross entropy
    _, class_count = np.unique(eeg_dataset["train_label"], return_counts=True)
    weights = torch.tensor((sum(class_count)/class_count)**args.booster, dtype=torch.float)
    from sklearn.utils.class_weight import compute_class_weight
    weights = torch.tensor(compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1, 2, 3, 4]),
        y=eeg_dataset["train_label"]
    )**args.booster, dtype=torch.float)
    
def conv1d_output_length(L_in, kernel, stride, padding=0, dilation=1):
    return math.floor((L_in + 2*padding - dilation*(kernel-1) - 1)/stride + 1)

def compute_mirror_deconv(conv_kernels, conv_strides, input_length):
    """
    Compute CNN output lengths and deconv params to mirror CNN.
    Ensures all kernels are positive.
    """
    # 1️⃣ Compute CNN output lengths
    cnn_lengths = []
    L = input_length
    conv_paddings = [0,0,0]
    for k, s, p in zip(conv_kernels, conv_strides, conv_paddings):
        L = conv1d_output_length(L, k, s, p)
        cnn_lengths.append(L)
    deconv_params = []
    # Build list of deconv target lengths: each deconv output = previous CNN layer output
    targets = [input_length] + cnn_lengths[:-1]  # last deconv expands to original input
    deconv_kernel = []
    deconv_stride = []
    for L_in, stride, target_L in zip(reversed(cnn_lengths), reversed(conv_strides), reversed(targets)):
        kernel = target_L - (L_in - 1) * stride
        while kernel <= 0 and stride > 0:
            stride -=1
            kernel = target_L - (L_in - 1) * stride
        deconv_kernel.append(kernel)
        deconv_stride.append(stride)

    return tuple(deconv_kernel), tuple(deconv_stride)

print(f"Loading model... {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
# randomly initialized model

if 'none' in args.model_pathway or 'r_d' in args.model_pathway or 'pd' in args.model_pathway:
    from configuration import Config 
    config = Config()
    freq = 250
    data_len = 30
    if 'sleep_edf' in args.train_path or 'r_d' in args.model_pathway:
        config.conv_kernel = (100,112,40)
        config.conv_stride = (2,2,2)
        freq = 100

    if args.mission == "next":
        config.deconv_kernel, config.deconv_stride = compute_mirror_deconv(config.conv_kernel, config.conv_stride,
                                                                           eeg_dataset['train'][0]["input_values"].size)
        data_len = args.data_length   
    if args.feature_ext == 'stft':
        config.max_source_positions = len(trainy[0]['input_values'])
        # if data_len == 60:
        #     config.max_source_positions = 59 
        # config.max_target_positions = 448
        config.mask_time_length=1 
        config.conv_dim = (251,251,251)
    else:
        L = data_len * freq
        for k, s in zip(config.conv_kernel, config.conv_stride): L = conv1d_output_length(L, k, s, 0)
        config.max_source_positions = L
    model = Wav2Vec2ForSequenceClassification(config)
    if 'pd' in args.model_pathway:
        config.decoder_layers = 3
        config.encoder_layers = 4
        model = WhisperModel(config)
# pretrained model
else:
    pretrained_model_path = (args.model_pathway)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(pretrained_model_path)
    config = model.config
    print('config', config)
print(f"Loaded model {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print('max_source_positions',  config.max_source_positions)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
data_collator = DataCollatorForWav2Vec2Pretraining(
    model= model,
    feature_extractor=feature_extractor,
    pad_to_multiple_of=None,
    mask_time_prob=config.mask_time_prob,
    mask_time_length=config.mask_time_length, # should be around 5s, consider the whole seq to be 30 seq
    has_cnn = not args.feature_ext == 'stft'
)

train_dataloader = DataLoader(eeg_dataset["train"], batch_size=args.per_device_train_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=data_collator)
validation_dataloader = DataLoader(eeg_dataset["validation"], batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=data_collator)
test_dataloader = DataLoader(eeg_dataset["test"], batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=data_collator)

if args.encoder_sit == 'frozen_enc' or args.encoder_sit == 'cnn_only' or 'pd' in args.model_pathway:
    optimizer = AdamW(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=[args.adam_beta1, args.adam_beta2],
        eps=args.adam_epsilon,
    )
else:
    optimizer = AdamW(
        [
            {
                "params": list(model.model.encoder.parameters()) +
                          list(model.model.decoder.parameters()),
                "lr": args.learning_rate_t  
            },
            {
                "params": list(model.model.feature_extractor.parameters()) +
                          list(model.projector.parameters()) +
                          list(model.classifier.parameters()),
                "lr": args.learning_rate  
            }
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
# Prepare everything with our `accelerator`.

accelerator = Accelerator(mixed_precision="fp16")
print('accelerator initialized', flush=True)
print(accelerator.state, flush=True)
print(accelerator.is_local_main_process, flush=True)
logger.info(accelerator.state, main_process_only=True)
if accelerator.is_local_main_process:
    #datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
accelerator.wait_for_everyone()

model, optimizer, train_dataloader, validation_dataloader, test_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, validation_dataloader, test_dataloader
)
print(next(model.parameters()).device)
print(torch.cuda.is_available())

# Scheduler and math around the number of training steps.
print(f"dataloader length: {len(train_dataloader)}", flush=True)
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
num_train_epochs = args.num_train_epochs
max_train_steps = num_train_epochs * num_update_steps_per_epoch
p_max_train_steps = 100 * num_update_steps_per_epoch
print(f"num_update_steps_per_epoch: {num_update_steps_per_epoch}", flush=True)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=max_train_steps,
)

from torch.optim.lr_scheduler import LambdaLR

def general_model_lr_schedule(step):
    if step <= 100000:
        return 1e-3
    elif step <= 200000:
        return 1e-4
    else:
        return 1e-5

def encoder_lr_schedule(step):
    return 1e-6  # always constant

# Schedule returns a multiplier for the initial lr
#if args.encoder_sit == 'frozen_enc' or args.encoder_sit == 'cnn_only':
#    lr_scheduler = LambdaLR(
#        optimizer,
#        lr_lambda=[
#            lambda step: general_model_lr_schedule(step) / args.learning_rate
#        ]
#    )
#else:
#    lr_scheduler = LambdaLR(
#        optimizer,
#        lr_lambda=[
#            lambda step: encoder_lr_schedule(step) / args.learning_rate_t,
#            lambda step: general_model_lr_schedule(step) / args.learning_rate
#        ]
#    )

# Afterwards we recalculate our number of training epochs
#num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

print(f"num_train_epochs: {num_train_epochs}", flush=True)
def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)
def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

# ---- 5. Train
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['max_split_size_mb'] = '512'

total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

logger.info("***** Running training *****")
logger.info(f"  Num Epochs = {num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {max_train_steps}")
completed_steps = 0
starting_epoch = 0
# Only show the progress bar once on each machine.
progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
starting_epoch = 0


model = model.cuda()
# if not args.feature_ext == 'cnn_frozen':
#     model.initialize_feature_encoder()

print(f"Starting the training loop {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

val_log = {}
val_log[f"val_f1_weighted_{patient_name}"] =[]

test_log = {}
test_log[f"test_f1_weighted_{patient_name}"] =[]
test_log[f"test_f1_macro_{patient_name}"] =[]
test_log[f"test_ck_{patient_name}"] =[]
test_predict = []

RESULTS_DIR = Path("/home/mengkris/links/scratch/results")

def init_metrics(version):
    """Delete existing metric files for a version before training starts."""
    for suffix in ["_val_metrics.csv", "_test_metrics.csv"]:
        path = RESULTS_DIR / f"{version}{suffix}"
        if path.exists():
            path.unlink()  # delete the old file
            print(f"Removed old file: {path}")

def save_metrics(version, epoch, val_metrics=None, test_metrics=None):
    """
    Save validation or test metrics to CSV files named after the model version.

    version: str
        Experiment or model version (used for filenames)
    epoch: int or str
        Current epoch for validation, or 'final' for test
    val_metrics / test_metrics: dict
        Dictionary of metrics (pass one of them per call)
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if val_metrics is not None:
        val_path = RESULTS_DIR / f"{version}_val_metrics.csv"
        val_df = pd.DataFrame([{**{"epoch": epoch}, **val_metrics}])
        val_df.to_csv(val_path, mode="a", header=not os.path.exists(val_path), index=False)

    if test_metrics is not None:
        test_path = RESULTS_DIR / f"{version}_test_metrics.csv"
        test_df = pd.DataFrame([{**{"epoch": epoch}, **test_metrics}])
        test_df.to_csv(test_path, mode="a", header=not os.path.exists(test_path), index=False)

init_metrics(args.vers)
output_dir = f"/home/mengkris/links/scratch/pretrained_{patient_name}_{args.vers}"
if 'pd' in args.model_pathway and not os.path.exists(output_dir):
    print('Starting pd pretraining from scratch', flush=True)
    for epoch in range(starting_epoch, 100): #100
        _ = gc.collect()
        model = model.cuda()
        model.train()
        for step, batch in enumerate(train_dataloader):
            num_losses = batch["mask_time_indices"].sum()
            batch['has_decoder'] = True
            batch['mission'] = args.mission
            if args.feature_ext == 'stft':
                batch['stft'] = True
                batch['attention_mask'] = None
                batch['sub_attention_mask'] = None
            batch['task'] = 'reconstruction_l1'
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()
            found_bad_tensor=False
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device='cuda')
                    if torch.isnan(batch[k]).any() or torch.isinf(batch[k]).any():
                        found_bad_tensor = True
                        print(f"[SKIP] NaN or Inf in batch at step {step}, key = {k}")
                        break
            if found_bad_tensor:
                continue  # Skip this batch

            ## forward
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if accelerator.state.num_processes > 1:
                num_losses = accelerator.gather_for_metrics(num_losses).sum()
                gradient_multiplier = accelerator.state.num_processes / num_losses
                multiply_grads(model.module.parameters(), gradient_multiplier)
            else:
                multiply_grads(model.parameters(), 1)
                multiply_grads(model.parameters(), 1 / num_losses)

            ## update step
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # compute grad norm for monitoring
                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1
                )
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.parameters(), scale)

                # update parameters
                optimizer.step()
                optimizer.zero_grad()
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )

                progress_bar.update(1)
                completed_steps += 1

            # 6. Log all results
            if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
                loss.detach()
                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather_for_metrics(loss).sum()
                    percent_masked = accelerator.gather_for_metrics(percent_masked).sum()###

                train_logs = {
                    "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                    "%_mask_idx": percent_masked / accelerator.num_processes,###
                    "lr": torch.tensor(optimizer.param_groups[0]["lr"]),###
                    "grad_norm": torch.tensor(grad_norm),###
                }

                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item() if hasattr(v, "item") else v)

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)
                    if is_wandb_available():
                        wandb.log(train_logs)

            # save model every `args.saving_steps` steps
            if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
                if (args.push_to_hub and epoch < args.num_train_epochs - 1) or output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )


            # if completed steps > `args.max_train_steps` stop

            if completed_steps >= p_max_train_steps:
                torch.cuda.empty_cache()
                break

            torch.cuda.empty_cache()
    print('Pretraining step done', flush=True)
    print(os.path.exists(output_dir), flush=True)  
    print('pd' in args.model_pathway, flush=True) 
if 'pd' in args.model_pathway and os.path.exists(output_dir):
    print('Loading pd pretrained model', flush=True)
    pretrained_model_path = (output_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(pretrained_model_path)
    config = model.config
    optimizer = AdamW(
        [
            {
                "params": list(model.model.encoder.parameters()) +
                          list(model.model.decoder.parameters()),
                "lr": args.learning_rate_t  
            },
            {
                "params": list(model.model.feature_extractor.parameters()) +
                          list(model.projector.parameters()) +
                          list(model.classifier.parameters()),
                "lr": args.learning_rate  
            }
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
    model, optimizer, train_dataloader, validation_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, test_dataloader
    )
    model = model.cuda()


completed_steps = 0
starting_epoch = 0
best_f1 = 0.0
if not args.feature_ext == 'cnn_frozen':
    model.initialize_feature_encoder()

for epoch in range(starting_epoch, num_train_epochs):

    train_logs = {}
    train_logs["train_loss"]=[]
    train_logs["train_w_f1"]=[]

    _ = gc.collect()
    if args.feature_ext == 'cnn_frozen':
        model.freeze_feature_encoder()
    # elif args.feature_ext == 'window_shifting':
    # do something here 
    if args.encoder_sit == 'frozen_enc':
        model.freeze_encoder()
    if args.freeze_decoder:
        model.freeze_decoder()
    model.train()
    t_true = []
    t_predict = []
    t_f1 = []
    for step, batch in enumerate(train_dataloader):
        num_losses = batch["mask_time_indices"].sum()
        batch['cnn_only'] = args.encoder_sit == 'cnn_only'
        batch['has_decoder'] = args.has_decoder
        if 'custom' in args.loss_type:
                batch['sub_task'] = 'custom_ce'
        if args.feature_ext == 'stft':
            batch["stft"] = True
            batch['attention_mask'] = None
            batch['sub_attention_mask'] = None
        if args.has_decoder and 'ce' in args.loss_type:
            batch['task'] = 'cross_entropy'
        if args.encoder_sit == 'unmasked_enc' or args.mission == 'next':
            batch['unmasked'] = True
            batch['attention_mask'] = None
        if args.weights:
            batch['weights'] = weights
        g = eeg_dataset["train_label"][step * total_batch_size:(step + 1) * total_batch_size]
        g = torch.tensor(g)
        #else: # later on deal with this coz this is for ctc, we might have a third need with mamba
        #    g = torch.tensor([item['input_labels'] for item in g])
        batch['labels'] = g
        t_true.append(g.detach().cpu())
        sub_attention_mask = batch.pop("sub_attention_mask", None) # might need some modifications if i am doing unmkased
        sub_attention_mask = (
            sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
        )
        percent_masked = num_losses / sub_attention_mask.sum()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device='cuda')

        ## forward
        outputs = model(**batch)
        loss = outputs.loss / args.gradient_accumulation_steps
        logits = outputs.logits
        print(batch['labels'])
        predictions = torch.argmax(logits, dim=-1) # might need modifications for ctc or mamba
        print(predictions)
        t_predict.append(predictions.detach().cpu())
        f1 = f1_score(g.cpu().numpy(), predictions.cpu().numpy(), average="weighted") # later on use another accuracy measurement, the one that MUSE guy talked about 
        t_f1.append(f1)
        accelerator.backward(loss)
        train_logs["train_loss"].append(outputs.loss.item())
        train_logs["train_w_f1"].append(f1)

        if accelerator.state.num_processes > 1:
            num_losses = accelerator.gather_for_metrics(num_losses).sum()
            gradient_multiplier = accelerator.state.num_processes / num_losses
            multiply_grads(model.module.parameters(), gradient_multiplier)
        else:
            multiply_grads(model.parameters(), 1)
            multiply_grads(model.parameters(), 1 / num_losses)

        ## update step
        if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            # compute grad norm for monitoring
            scale = (
                accelerator.scaler._scale.item()
                if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                else 1
            )
            if accelerator.state.num_processes > 1:
                grad_norm = get_grad_norm(model.module.parameters(), scale)
            else:
                grad_norm = get_grad_norm(model.parameters(), scale)

            # update parameters
            optimizer.step()
            optimizer.zero_grad()
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step()
            elif accelerator.is_local_main_process:
                progress_bar.write(
                    f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                )

            progress_bar.update(1)
            completed_steps += 1

        # 6. Log all results
        if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
            loss.detach()
            if accelerator.state.num_processes > 1:
                loss = accelerator.gather_for_metrics(loss).sum()
                percent_masked = accelerator.gather_for_metrics(percent_masked).sum()###

            train_log = {
                "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                "%_mask_idx": percent_masked / accelerator.num_processes,###
                "lr": torch.tensor(optimizer.param_groups[0]["lr"]),###
                "grad_norm": torch.tensor(grad_norm),###
                "accuracy": f1,
            }

            log_str = ""
            for k, v in train_log.items():
                log_str += "| {}: {:.3e}".format(k, v.item() if hasattr(v, "item") else v)

            if accelerator.is_local_main_process:
                progress_bar.write(log_str)
                if is_wandb_available():
                    wandb.log(train_log)

            # Print CUDA memory summary every 100 steps
            if (step + 1) % 100 == 0 and torch.cuda.is_available():
                print("\n[CUDA MEMORY SUMMARY - TRAINING]")
                print(torch.cuda.memory_summary())
        # save model every `args.saving_steps` steps
        # if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
        #     if (args.push_to_hub and epoch < args.num_train_epochs - 1) or args.output_dir is not None:
        #         accelerator.wait_for_everyone()
        #         unwrapped_model = accelerator.unwrap_model(model)
        #         unwrapped_model.save_pretrained(
        #             args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        #         )


        # if completed steps > `args.max_train_steps` stop

        if completed_steps >= max_train_steps:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            break
        # Per-batch cleanup to prevent memory leaks
        del outputs, logits, predictions, loss, batch
        #gc.collect()
        #torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        save_dir = os.path.join(args.output_dir, f"checkpoint-epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_model.save_pretrained(
            save_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
    # After loop: compute median
    median_loss = median(train_logs['train_loss'])
    median_accuracy = median(train_logs['train_w_f1'])
    #median_ck = median(train_logs['train_ck'])
    train_logs_median = {
        f"train_loss_median_{patient_name}": median_loss,
        f"train_accuracy_median_{patient_name}": median_accuracy,
        #f"train_ck_median_{patient_name}": median_ck
    }
    if is_wandb_available():
        wandb.log(train_logs_median)
    del median_loss, median_accuracy, train_logs_median, train_logs

    y_predict = []
    val_logs = {}
    val_logs[f"val_f1_weighted_{patient_name}"] = []
    val_logs[f"val_f1_macro_{patient_name}"] = []
    val_logs[f"val_ck_{patient_name}"] = []
    model.eval()
    for step, batch in enumerate(validation_dataloader):
        with torch.no_grad():
            
            batch.pop("sub_attention_mask", None)
            batch['cnn_only'] = args.encoder_sit == 'cnn_only'
            batch['has_decoder'] = args.has_decoder
            if 'custom' in args.loss_type:
                    batch['sub_task'] = 'custom_ce'
            if args.feature_ext == 'stft':
                batch["stft"] = True
                batch['attention_mask'] = None
            if args.has_decoder and 'ce' in args.loss_type:
                batch['task'] = 'cross_entropy'
            #if args.encoder_sit == 'unmasked_enc':
            batch['unmasked'] = True
            if batch['unmasked']:
                batch['attention_mask'] = None
            if args.weights:
                batch['weights'] = weights
            j = eeg_dataset["validation_label"][step * total_batch_size:(step + 1) * total_batch_size]
            
            j = torch.tensor(j)
            #else:
            #    j = torch.tensor([item['input_labels'] for item in j])
            batch['labels'] = j
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device='cuda')
           

            output = model(**batch)
            logits = output.logits
            print(batch['labels'])
            predictions = torch.argmax(logits, dim=-1)
            print('validation pred:', predictions)
            y_predict.append(predictions.detach().cpu())

            # Print CUDA memory summary every 100 steps
            if (step + 1) % 100 == 0 and torch.cuda.is_available():
                print("\n[CUDA MEMORY SUMMARY - VALIDATION]")
                print(torch.cuda.memory_summary())
            # Per-batch cleanup to prevent memory leaks
            del output, logits, predictions, batch
            #gc.collect()
            #torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    f1_w = f1_score(eeg_dataset["validation_label"], torch.cat(y_predict).cpu().numpy(), average="weighted")
    f1_m = f1_score(eeg_dataset["validation_label"], torch.cat(y_predict).cpu().numpy(), average="macro")
    f1_per_class = f1_score(eeg_dataset["validation_label"], torch.cat(y_predict).cpu().numpy(), average=None, labels=[0,1,2,3,4])
    print(f"F1 per class: {f1_per_class}", flush=True)
    ck = cohen_kappa_score(eeg_dataset["validation_label"], torch.cat(y_predict).cpu().numpy(), labels=[0, 1, 2, 3, 4])
    val_metrics = {
        "val_f1_weighted": f1_w,
        "val_f1_macro": f1_m,
        "val_f1_W": f1_per_class[0],
        "val_f1_N1": f1_per_class[1],
        "val_f1_N2": f1_per_class[2],
        "val_f1_N3": f1_per_class[3],
        "val_f1_R": f1_per_class[4],
        "val_ck": ck,
    }
    if val_metrics["val_f1_weighted"] > best_f1:
        best_f1 = val_metrics["val_f1_weighted"]
        best_epoch = epoch
        print(best_epoch, flush=True)
        # Save labels only for best epoch
        best_val_metrics = {**val_metrics,
                            "val_predictions": torch.cat(y_predict).cpu().tolist(),
                            "val_true_labels": eeg_dataset["validation_label"].tolist()}
    #save_metrics(args.vers, epoch, val_metrics=val_metrics)
    val_logs[f"val_f1_weighted_{patient_name}"]=f1_w
    val_logs[f"val_f1_macro_{patient_name}"]=f1_m
    val_logs[f"val_ck_{patient_name}"]=ck

    val_log[f"val_f1_weighted_{patient_name}"].append(f1_w)


    log_str = ""
    for k, v in val_logs.items():
        log_str += "| {}: {:.3e}".format(k, v.item() if hasattr(v, "item") else v)

    if accelerator.is_local_main_process:
        progress_bar.write(log_str)
        
    if is_wandb_available():
        wandb.log(val_logs)


    del f1_w, f1_m, ck, f1_per_class
    gc.collect()
    torch.cuda.empty_cache()
    ## Label names corresponding to 0,1,2,3,4
    #labels = ['W', 'N1', 'N2', 'N3', 'R']
    ## Compute confusion matrix
    #y_true_flat = eeg_dataset["validation_label"]
    #y_predict_flat = torch.cat(y_predict).cpu().numpy()
    #cm = confusion_matrix(y_true_flat, y_predict_flat , normalize='true')
    ## Plot confusion matrix
    #plt.figure(figsize=(6, 5))
    #sns.heatmap(cm, annot=True, fmt=".2f", cmap='Greys',
    #            xticklabels=labels, yticklabels=labels,
    #            vmin=0, vmax=1)

    #plt.xlabel('Predicted Label')
    #plt.ylabel('True Label')
    #plt.title(f'{patient_name}, Epoch: {epoch} Val Acc Mean: {np.mean(y_true_flat == y_predict_flat):.3f}') #F1 Median: {y_f1_flat}
    #plt.tight_layout()

    ## Save the plot
    #plt.savefig(f'/home/mengkris/links/scratch/ehb_runs/den_{epoch}_{args.vers}_val_matrix.png', dpi=300)
    #plt.close()
    y_predict.clear()
    del val_metrics, val_logs, y_predict # ,cm

best_epoch = val_log[f"val_f1_weighted_{patient_name}"].index(max(val_log[f"val_f1_weighted_{patient_name}"]))
print(best_epoch, max(val_log[f"val_f1_weighted_{patient_name}"]), flush = True)
save_metrics(args.vers, best_epoch, val_metrics=best_val_metrics)
best_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch_{best_epoch}")
best_model = Wav2Vec2ForSequenceClassification.from_pretrained(best_checkpoint_dir)
best_model.to('cuda')
best_model.eval()
for step, batch in enumerate(test_dataloader):
    with torch.no_grad():
        batch.pop("sub_attention_mask", None)
        batch['cnn_only'] = args.encoder_sit == 'cnn_only'
        batch['has_decoder'] = args.has_decoder
        if 'custom' in args.loss_type:
            batch['sub_task'] = 'custom_ce'
        if args.feature_ext == 'stft':
            batch["stft"] = True
            batch['attention_mask'] = None
        if args.has_decoder and 'ce' in args.loss_type:
            batch['task'] = 'cross_entropy'
        #if args.encoder_sit == 'unmasked_enc':
        batch['unmasked'] = True
        if batch['unmasked']:
            batch['attention_mask'] = None
        if args.weights:
            batch['weights'] = weights
        h = eeg_dataset["test_label"][step * total_batch_size:(step + 1) * total_batch_size]
        
        h = torch.tensor(h)
        #else:
        #    h = torch.tensor([item['input_labels'] for item in h])
        batch['labels'] = h
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device='cuda')
        print(batch['labels'], flush=True)
        output = best_model(**batch)
        logits = output.logits
        predictions = torch.argmax(logits, dim=-1)
        print('test pred:', predictions)
        test_predict.append(predictions)
f1_w = f1_score(eeg_dataset["test_label"], torch.cat(test_predict).cpu().numpy(), average="weighted")
f1_m = f1_score(eeg_dataset["test_label"], torch.cat(test_predict).cpu().numpy(), average="macro")
f1_per_class = f1_score(eeg_dataset["test_label"], torch.cat(test_predict).cpu().numpy(), average=None, labels=[0,1,2,3,4])
print(f"F1 per class: {f1_per_class}", flush=True)
ck = cohen_kappa_score(eeg_dataset["test_label"], torch.cat(test_predict).cpu().numpy(), labels=[0, 1, 2, 3, 4])
def get_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        return "unknown"
commit_hash = get_commit_hash()
test_metrics = {
    "test_f1_weighted": f1_w,
    "test_f1_macro": f1_m,
    "test_ck": ck,
    "test_f1_W": f1_per_class[0],
    "test_f1_N1": f1_per_class[1],
    "test_f1_N2": f1_per_class[2],
    "test_f1_N3": f1_per_class[3],
    "test_f1_R": f1_per_class[4],
    "test_pred": torch.cat(test_predict).cpu().tolist(),
    "true_label": eeg_dataset["test_label"].tolist(),
    "git_version":commit_hash
}
save_metrics(args.vers, best_epoch, test_metrics=test_metrics)

test_log[f"test_f1_weighted_{patient_name}"]=f1_w
test_log[f"test_f1_macro_{patient_name}"]=f1_m
test_log[f"test_ck_{patient_name}"]=ck

log_str = ""
for k, v in test_log.items():
    log_str += "| {}: {:.3e}".format(k, v.item() if hasattr(v, "item") else v)

if accelerator.is_local_main_process:
    progress_bar.write(log_str)
if is_wandb_available():
    wandb.log(test_log)
del f1_w, f1_m, ck, f1_per_class, test_log
gc.collect()
torch.cuda.empty_cache()

#test_true_flat = eeg_dataset["test_label"]
#test_predict_flat = torch.cat(test_predict).cpu().numpy()

#cm = confusion_matrix(test_true_flat, test_predict_flat , normalize='true')
## Plot confusion matrix
#plt.figure(figsize=(6, 5))
#sns.heatmap(cm, annot=True, fmt=".2f", cmap='Greys',
#            xticklabels=labels, yticklabels=labels,
#            vmin=0, vmax=1)

#plt.xlabel('Predicted Label')
#plt.ylabel('True Label')
#plt.title(f'{patient_name}, Epoch: {best_epoch} Test Acc Mean: {np.mean(test_true_flat == test_predict_flat):.3f}') #F1 Median: {y_f1_flat}
#plt.tight_layout()

# Save the plot
#plt.savefig(f'/home/mengkris/links/scratch/ehb_runs/TEST_{best_epoch}_{args.vers}_test_matrix.png', dpi=300)
#plt.close()


#del cm, test_predict



