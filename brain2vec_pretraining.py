import gc
import math
import argparse
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from collections import namedtuple, defaultdict
import transformers
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from typing import Optional, Union, Tuple
from transformer_encoder_decoder import WhisperModel, _compute_mask_indices, WhisperPreTrainedModel, _compute_last_fixed_mask
from configuration import Config
import wandb
from accelerate.logging import get_logger
from transformers import (
    AdamW,
    SchedulerType,
    Wav2Vec2FeatureExtractor,
    get_scheduler,
    is_wandb_available,
    set_seed,
)

#wandb.login(key="781ffa93f4f0316f1be9b044cb3302b5bfb87af7")
logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--num_warmup_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_duration_in_seconds", type=float, default=10000.0)
    parser.add_argument("--min_duration_in_seconds", type=float, default=2.0)
    parser.add_argument("--has_decoder", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--task", type=str, default="reconstruction_l1")
    parser.add_argument("--mission", type=str, default="now")
    parser.add_argument("--feature_ext", type=str, default="cnn")
    parser.add_argument("--decoder_type", type=str, default="none")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--saving_steps", type=int, default=500)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_test_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--mask_time_prob", type=float, default=0.65)
    parser.add_argument("--mask_time_length", type=int, default=50)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--validation_split_percentage", type=int, default=10)
    parser.add_argument("--data_type", type=str, default="ecog")
    parser.add_argument("--train_cache_file_name", type=str, default=None)
    parser.add_argument("--validation_cache_file_name", type=str, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push_to_hub", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--data_length", type=int, default=30)
    parser.add_argument("--mask_length", type=int, default=0)
    parser.add_argument("--notes", type=str, default="")

    return parser.parse_args()


args = parse_args()
if args.mask_length > 0:
    args.output_dir = f"{args.output_dir}_{args.data_type}_{args.task}_{args.mission}_{args.feature_ext}_{args.data_length}_{args.mask_length}"
else:
    args.output_dir = f"{args.output_dir}_{args.data_type}_{args.task}_{args.mission}_{args.feature_ext}"
tags = ['pretraining', args.task, args.data_type, args.mission]
if args.feature_ext == 'stft':
    tags.append("stft")
if args.mask_length != 0:
    tags.append(f"mask{args.mask_length}")
if args.data_length != 30:
    tags.append(f"{args.data_length}sec")
version = args.data_type + '_' + args.task + '_' + args.mission + '_' + str(args.data_length) + '_' + str(args.mask_length)
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


accelerator = Accelerator()
logger.info(accelerator.state, main_process_only=True)
if accelerator.is_local_main_process:
    #datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
accelerator.wait_for_everyone()
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

    model: WhisperModel
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10
    mask_length: Optional[int] = 0
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

        if self.mask_length != 0:
            features_shape = (batch_size, batch["input_values"].shape[-1] )
            frames_forecast = self.mask_length * batch["input_values"].shape[-1] // (args.data_length+self.mask_length)
            if args.feature_ext == 'stft':
                features_shape = (batch_size, batch["input_values"].shape[-2])
                frames_forecast = self.mask_length * batch["input_values"].shape[-2] // (
                            args.data_length + self.mask_length)
            batch["mask_length"] = torch.tensor(frames_forecast)
            mask_time_indices = _compute_last_fixed_mask(
                features_shape,
                frames_to_mask=frames_forecast,
            )
        else:
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

dataset = {}
train_y = None
test_y = None
# loading the ecog data
if args.data_type == 'ecog':
    ecog_data = np.load('/home/mengkris/links/scratch/norm_ecog_data.npy', allow_pickle=True)
    ind_ch_ecog_data = []
    for arr in ecog_data:
        for ch in arr['input_values']:
            ind_ch_ecog_data.append({'input_values': ch})
    ind_ch_ecog_data = np.array(ind_ch_ecog_data)
    train_x, test_x = train_test_split(ind_ch_ecog_data, test_size=0.10)
# loading the eeg headband data
# loading the ecog data
if args.data_type == 'sleep_edf':
    # # Path to your folder
    # DATA_DIR = "/home/kristal/Desktop/casette_eeg"
    #
    # # Load all subject files
    # all_samples = []
    # for fname in os.listdir(DATA_DIR):
    #     if fname.endswith(".npy"):
    #         print(fname)
    #         fpath = os.path.join(DATA_DIR, fname)
    #         subject_data = np.load(fpath, allow_pickle=True).item()
    #
    #         # subject_data should be a dict with keys: "subject_id", "labels", "fpz", "pz"
    #         sid = subject_data["subject_id"]
    #         labels = subject_data["labels"]
    #         fpz = subject_data["fpz"]
    #         pz = subject_data["pz"]
    #
    #         for i in range(len(labels)):
    #             sample = {
    #                 "pid": sid,
    #                 "label": int(labels[i]),
    #                 "fpz": fpz[i],
    #                 "pz": pz[i]
    #             }
    #             all_samples.append(sample)
    #
    # # -------- Group samples by patient --------
    # patient_to_samples = defaultdict(list)
    # for sample in all_samples:
    #     patient_to_samples[sample["pid"]].append(sample)
    #
    # # Target for training
    # total_samples = len(all_samples)
    # train_target = int(0.85 * total_samples)
    #
    # # Sort patients by number of samples (largest first)
    # patients_sorted = sorted(patient_to_samples.items(), key=lambda x: len(x[1]), reverse=True)
    #
    # # Greedy split
    # train_data, test_data = [], []
    # train_count = 0
    #
    # for pid, samples in patients_sorted:
    #     if train_count + len(samples) <= train_target:
    #         train_data.extend(samples)
    #         train_count += len(samples)
    #     else:
    #         test_data.extend(samples)
    #
    #
    #
    # # -------- Split EEG channels --------
    # def split_by_channel(data):
    #     input_values, labels, pids = [], [], []
    #     for sample in data:
    #         # fpz as a separate instance
    #         input_values.append({'input_values':sample["fpz"]})  # shape (1, timepoints)
    #         labels.append(sample["label"])
    #         pids.append(sample["pid"])
    #
    #         # pz as a separate instance
    #         input_values.append({'input_values':sample["pz"]})  # shape (1, timepoints)
    #         labels.append(sample["label"])
    #         pids.append(sample["pid"])
    #
    #     return np.array(input_values), np.array(labels), np.array(pids)
    #
    #
    # train_x, train_y, train_p = split_by_channel(train_data)
    # test_x, test_y, test_p = split_by_channel(test_data)

    if args.mission == 'next':
        train_x = np.load('/home/mengkris/links/scratch/casette_eeg/N_sleep_edf_train_values.npy', allow_pickle=True)
        train_y = np.load('/home/mengkris/links/scratch/casette_eeg/N_sleep_edf_train_labels.npy', allow_pickle=True)
        train_p = np.load('/home/mengkris/links/scratch/casette_eeg/N_sleep_edf_train_pid.npy', allow_pickle=True)
        test_x = np.load('/home/mengkris/links/scratch/casette_eeg/N_sleep_edf_test_values.npy', allow_pickle=True)
        test_y = np.load('/home/mengkris/links/scratch/casette_eeg/N_sleep_edf_test_labels.npy', allow_pickle=True)
        test_p = np.load('/home/mengkris/links/scratch/casette_eeg/N_sleep_edf_test_pid.npy', allow_pickle=True)
    else:
        train_x = np.load('/home/mengkris/links/scratch/casette_eeg/sleep_edf_train_values.npy', allow_pickle=True)
        train_y = np.load('/home/mengkris/links/scratch/casette_eeg/sleep_edf_train_labels.npy', allow_pickle=True)
        train_p = np.load('/home/mengkris/links/scratch/casette_eeg/sleep_edf_train_pid.npy', allow_pickle=True)
        test_x = np.load('/home/mengkris/links/scratch/casette_eeg/sleep_edf_test_values.npy', allow_pickle=True)
        test_y = np.load('/home/mengkris/links/scratch/casette_eeg/sleep_edf_test_labels.npy', allow_pickle=True)
        test_p = np.load('/home/mengkris/links/scratch/casette_eeg/sleep_edf_test_pid.npy', allow_pickle=True)
    # # Save each part separately
    # np.save('/home/kristal/Desktop/casette_eeg/sleep_edf_train_values.npy', train_x)
    # np.save('/home/kristal/Desktop/casette_eeg/sleep_edf_train_labels.npy', train_y)
    # np.save('/home/kristal/Desktop/casette_eeg/sleep_edf_train_pid.npy', train_p)
    #
    # np.save('/home/kristal/Desktop/casette_eeg/sleep_edf_test_values.npy', test_x)
    # np.save('/home/kristal/Desktop/casette_eeg/sleep_edf_test_labels.npy', test_y)
    # np.save('/home/kristal/Desktop/casette_eeg/sleep_edf_test_pid.npy', test_p)


# loading pd data
elif args.data_type == 'pd': #gotta correct this later on coz the DIRECTERIES ARE OFF
    train = np.load('/home/mengkris/links/scratch/pd_data/all_but_one_patient_zscored_train_lfp/1_STAN.npy', allow_pickle=True)
    train_label = np.load('/home/mengkris/links/scratch/pd_data/all_but_one_patient_zscored_train_label/1_STAN.npy', allow_pickle=True)
    validation = np.load('/home/mengkris/links/scratch/pd_data/one_patient_zscored_val_lfp/1_STAN.npy', allow_pickle=True)
    validation_label = np.load('/home/mengkris/links/scratch/pd_data/one_patient_zscored_val_label/1_STAN.npy', allow_pickle=True)
    train_x = np.array(train)
    train_y = np.array(train_label)
    test_x = np.array(validation)
    test_y = np.array(validation_label)

from scipy.signal import stft, get_window
def stft_transform(dataset, fs, mode='no-band', window_size=2, overlap=1):
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
    if args.data_type == 'sleep_edf':
        f = 100
    else:
        f = 250
    train_x = stft_transform(train_x, fs= f)
    test_x = stft_transform(test_x, fs = f)

if args.mission =='next':
    unmasked_shape = np.array(custom_segments(train_x[:10], args.data_length, None, window_size=30, key="input_values"))
    train_x = np.array(custom_segments(train_x, args.data_length, args.mask_length, window_size=30, key="input_values"))
    test_x = np.array(custom_segments(test_x, args.data_length , args.mask_length, window_size=30, key="input_values"))
    if train_y is not None:
        train_y = np.array(custom_segments(train_y, args.data_length, args.mask_length, window_size=30, key=None))
        test_y = np.array(custom_segments(test_y, args.data_length, args.mask_length, window_size=30, key=None))


dataset['train'], dataset["validation"] = train_x, test_x
if train_y is not None:
    dataset["train_label"], dataset["validation_label"] = train_y, test_y

# np.save(args.output_dir + '/train_data.npy', train_x)
# np.save(args.output_dir + '/train_labels.npy', train_y)
# np.save(args.output_dir + '/test_data.npy', test_x)
# np.save(args.output_dir + '/test_labels.npy', test_y)

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

# define feature extractor, config, and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('/home/mengkris/links/scratch/patrick_vonplaten_wav2vec2_model')
config = Config()
config.decoder_layers = args.decoder_layers
config.encoder_layers = args.encoder_layers
recording_freq = 250
if 'sleep_edf' in args.data_type:
        config.conv_kernel = (100,112,40)
        config.conv_stride = (2,2,2)
        recording_freq = 100
if args.feature_ext == 'stft':
        config.max_source_positions = len(unmasked_shape[0]['input_values'])
        config.mask_time_length=1 
        config.conv_dim = (int(recording_freq+1),int(recording_freq+1),int(recording_freq+1))

if args.mission == "next":
        config.deconv_kernel, config.deconv_stride = compute_mirror_deconv(config.conv_kernel, config.conv_stride, dataset['train'][0]["input_values"].size)
if args.feature_ext != 'stft':
    L = args.data_length * recording_freq
    for k,s in zip(config.conv_kernel, config.conv_stride): L = conv1d_output_length(L, k, s, 0)
    config.max_source_positions = L
        
model = WhisperModel(config)

# get masked data and attention masks
data_collator = DataCollatorForWav2Vec2Pretraining(
    model= model,
    feature_extractor=feature_extractor,
    pad_to_multiple_of=None,
    mask_time_prob=config.mask_time_prob,
    mask_time_length=config.mask_time_length, # should be around 5s, consider the whole seq to be 30 seq
    mask_length=args.mask_length,
    has_cnn = not args.feature_ext == 'stft'
)
# batch and shuffle data
train_dataloader = DataLoader(dataset["train"], batch_size=args.per_device_train_batch_size, shuffle=(args.data_type != 'pd'), collate_fn=data_collator)
validation_dataloader = DataLoader(dataset["validation"], batch_size=args.per_device_eval_batch_size, shuffle=(args.data_type != 'pd'), collate_fn=data_collator)

# setup optimizer
optimizer = AdamW(
    list(model.parameters()),
    lr=args.learning_rate,
    betas=[args.adam_beta1, args.adam_beta2],
    eps=args.adam_epsilon,
)
# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, validation_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, validation_dataloader
)

# Scheduler and math around the number of training steps.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
num_train_epochs = args.num_train_epochs
max_train_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=max_train_steps,
)

# Afterwards we recalculate our number of training epochs
#num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

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
reconstruction_loss = []


for epoch in range(starting_epoch, num_train_epochs):
    _ = gc.collect()
    print(epoch)
    model = model.cuda()
    model.train()
    for step, batch in enumerate(train_dataloader):
        num_losses = batch["mask_time_indices"].sum()
        batch['has_decoder'] = args.has_decoder
        batch['mission'] = args.mission
        if args.feature_ext == 'stft':
            batch['stft'] = True
            batch['attention_mask'] = None
            batch['sub_attention_mask'] = None
        if args.has_decoder and args.mission == 'next':
            batch['decoder_type'] = args.decoder_type
            if args.feature_ext == 'stft':
                batch['decoder_type'] = 'projection'
        batch['task'] =args.task
        if args.task == 'cross_entropy':
            j = dataset["train_label"][step * total_batch_size:(step + 1) * total_batch_size]
            batch['labels'] = torch.tensor(j)
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
        og_signal = outputs.feature_encoder_output[outputs.mask_time_indices]
        rec_signal = outputs.last_hidden_state[outputs.mask_time_indices]
        if args.task == 'cross_entropy':
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            f1 = f1_score(j, predictions.cpu().numpy(), average="weighted")
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
            if args.task == 'cross_entropy':
                train_logs["pret_accuracy"]=f1

            log_str = ""
            for k, v in train_logs.items():
                log_str += "| {}: {:.3e}".format(k, v.item() if hasattr(v, "item") else v)

            if accelerator.is_local_main_process:
                progress_bar.write(log_str)
                if is_wandb_available():
                    wandb.log(train_logs)

        # save model every `args.saving_steps` steps
        if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
            if (args.push_to_hub and epoch < args.num_train_epochs - 1) or args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )


        # if completed steps > `args.max_train_steps` stop

        if completed_steps >= max_train_steps:
            torch.cuda.empty_cache()
            break

        torch.cuda.empty_cache()
    if args.task != 'cross_entropy':
        stride = 50  # plot every 100th point for downsampling

        # Detach tensors from the computation graph and move to CPU
        og_np = og_signal.detach().cpu().numpy()[::stride]
        rec_np = rec_signal.detach().cpu().numpy()[::stride]

        # Plot
        plt.figure(figsize=(14, 5))
        plt.plot(og_np, label='Original Signal')
        plt.plot(rec_np, label='Reconstructed Signal', linestyle='--')
        plt.title("Original vs Reconstructed Signal at Masked Indices")
        plt.xlabel("Masked Time Index (downsampled)")
        plt.ylabel("Signal Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/home/mengkris/links/scratch/pretraining_figs/{version}_{epoch}_train.png', dpi=300)
        plt.close()
        del og_np, rec_np, og_signal, rec_signal


    model.eval()
    for step, batch in enumerate(validation_dataloader):
        with torch.no_grad():
            batch.pop("sub_attention_mask", None)
            batch['has_decoder'] = args.has_decoder
            batch['mission'] = args.mission
            if args.has_decoder and args.mission == 'next':
                batch['decoder_type'] = args.decoder_type
                if args.feature_ext == 'stft':
                    batch['decoder_type'] = 'projection'
            if args.feature_ext == 'stft':
                batch['stft'] = True
                batch['attention_mask'] = None
            batch['task'] = args.task
            if args.task == 'cross_entropy':
                j = dataset["validation_label"][step * total_batch_size:(step + 1) * total_batch_size]
                batch['labels'] = torch.tensor(j)
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(
                    batch["mask_time_indices"])
            )
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
            og_signalv = outputs.feature_encoder_output[outputs.mask_time_indices]
            rec_signalv = outputs.last_hidden_state[outputs.mask_time_indices]

            val_loss = outputs.loss / args.gradient_accumulation_steps
            if args.task == 'cross_entropy':
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                f1 = f1_score(j, predictions.cpu().numpy(), average="weighted")
                val_log = {"pretraining_val_loss": val_loss, "pretraining_val_acc": f1}
            else:
                val_log = {"pretraining_val_loss": val_loss}
            log_str = ""
            for k, v in val_log.items():
                log_str += "| {}: {:.3e}".format(k, v.item() if hasattr(v, "item") else v)
            if accelerator.is_local_main_process:
                progress_bar.write(log_str)
                if is_wandb_available():
                    wandb.log(val_log)
    if args.task != 'cross_entropy':
        og_np = og_signalv.detach().cpu().numpy()[::stride]
        rec_np = rec_signalv.detach().cpu().numpy()[::stride]

        # Plot
        plt.figure(figsize=(14, 5))
        plt.plot(og_np,label='Original Signal')
        plt.plot(rec_np, label='Reconstructed Signal', linestyle='--')
        plt.title("Original vs Reconstructed Signal at Masked Indices")
        plt.xlabel("Masked Time Index (downsampled)")
        plt.ylabel("Signal Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/home/mengkris/links/scratch/pretraining_figs/{version}_{epoch}_test.png', dpi=300)
        plt.close()
        del og_np, rec_np, og_signalv, rec_signalv

