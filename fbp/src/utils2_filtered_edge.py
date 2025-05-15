import os
import warnings
import tqdm
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.init as init
import obspy
import matplotlib.pyplot as plt
import seisbench.models as sbm  # noqa
from skimage import feature
# from denoising import langston_mousavi

from pathlib import Path


def is_nan(x):
    return x != x


def delete_nans(x: list):
    new_list = []
    for val in x:
        if is_nan(val) == False:
            new_list.append(val)

    return new_list


def gaussian_pick(onset, length, sigma):
    x = np.arange(length)
    if is_nan(onset) == True:
        return np.zeros(length)
    else:
        return np.exp(-np.power(x - onset, 2.0) / (2 * np.power(sigma, 2.0)))


def half_gaussian_pick(onset,
                       length,
                       sigma):
    gaussian = gaussian_pick(onset=onset,
                             length=length,
                             sigma=sigma)

    # Cut gaussian at its maximum value
    half_gaussian = gaussian[:np.argmax(gaussian)]
    ones = np.ones(int(length - len(half_gaussian)))

    return np.concatenate([half_gaussian, ones])


def heavyside(onset,
              length):
    if is_nan(onset) == True:
        return np.zeros(int(length))
    else:
        x0 = np.zeros(int(onset))
        x1 = np.ones(int(length - onset))

        return np.concatenate([x0, x1])


def boxcar(onset,
           length,
           width=20):
    if is_nan(onset) == True:
        return np.zeros(int(length))
    else:
        x0 = np.zeros(int(onset - width / 2))
        x1 = np.ones(int(width))
        x2 = np.zeros(int(length - onset - width / 2))

        return np.concatenate([x0, x1, x2])


def ramp_pick(onset,
              length,
              slope: float=1.0):
    if is_nan(onset) == True:
        return np.zeros(length)
    else:
        onset = int(onset)
        x = np.arange(length)
        x0 = np.zeros(onset)
        x_s = x[onset:int(1 / slope + onset)] * slope - onset * slope
        x1 = np.ones(len(x[int(1 / slope + onset):]))

        return np.concatenate([x0, x_s, x1])


class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(self, input, target):
        return nn.functional.cross_entropy(input=input,
                                           target=target)



class MeanSquaredError:
    def __init__(self):
        pass

    def __call__(self, input, target):
        """

        """
        mse = (input - target) ** 2
        mse = mse.mean(-1).sum(-1)
        mse = mse.mean()

        return mse


# class FocalLoss:
#     def __init__(self,
#                  alpha: float = 0.25,
#                  gamma: float = 2,
#                  reduction: str = "mean"):
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def __call__(self,
#                  input,
#                  target):
#         val =  torchvision.ops.sigmoid_focal_loss(inputs=input,
#                                                   targets=target,
#                                                   alpha=self.alpha,
#                                                   gamma=self.gamma,
#                                                   reduction=self.reduction)
#
#         if self.reduction == "none":
#             val = val.mean(-1).sum(-1)
#             val = val.mean()
#
#         return val


class DASDataset(Dataset):

    def __init__(self,
                 npz_files: list,
                 metadata_path: str,
                 shape: tuple=(1, 32, 2048),
                 sigma: int=20,
                 npz_data_key: str="data",
                 metadata_key: str="trace_P_arrival_sample",
                 norm: str="peak",
                 overlap: float=0.5,
                 ):
        self.npz_files = npz_files
        self.shape = shape
        self.metadata_path = metadata_path
        self.sigma = sigma
        self.npz_data_key = npz_data_key
        self.metadata_key = metadata_key
        self.norm = norm
        self.overlap = overlap

        # Determine overlap in samples, depending on output shape
        self.overlap_samples = int(shape[-1] * self.overlap)

        self._read_data()

    def _read_data(self):
        self.data = []
        for filename in self.npz_files:
            data = np.load(file=filename)[self.npz_data_key]

            # Read metadata
            metadata = pd.read_csv(os.path.join(self.metadata_path,
                                                f"metadata{Path(filename).stem}.csv")
                                   )

            # Reshape data to number of channels
            data = np.reshape(a=data,
                              newshape=(self.shape[0], *data.shape))

            resampled_data = np.empty(shape=(self.shape[0], data.shape[1], self.shape[-1]))

            # Create target output for each trace
            target = np.empty(shape=(self.shape[0], data.shape[1], self.shape[-1]))
            for idx in range(data.shape[1]):
                # Convert data to obspy Trace
                trace = obspy.Trace(data=data[0, idx, :], header=dict(sampling_rate=metadata.loc[idx, "sampling_rate"],
                                                                      starttime=metadata.loc[idx, "trace_start_time"],
                                                                      component="Z"))

                # Resampling trace to required sample length, ie self.shape[-1]
                sampling_rate_factor = data.shape[-1] / self.shape[-1]
                new_sampling_rate = metadata.loc[idx, "sampling_rate"] / sampling_rate_factor
                trace.resample(sampling_rate=new_sampling_rate)
                new_onset = np.ceil(metadata.loc[idx, self.metadata_key] / sampling_rate_factor)
                resampled_data[:, idx, :] = trace.data

                # target[:, idx, :] = gaussian_pick(onset=metadata.loc[idx, self.metadata_key],
                #                                     length=data.shape[2],
                #                                     sigma=self.sigma)
                # target[:, idx, :] = ramp_pick(onset=new_onset,
                #                               length=trace.stats.npts,
                #                               slope=0.05)
                target[:, idx, :] = heavyside(onset=new_onset,
                                              length=trace.stats.npts)
                # target[:, idx, :] = boxcar(onset=metadata.loc[idx, self.metadata_key],
                #                            length=data.shape[2],
                #                            width=self.sigma)

            data = resampled_data
            # Cut single chunks from entire data space
            # Loop over number of traces (outer loop)
            idx_start_out = 0
            idx_end_out = self.shape[1]
            step_out = int(self.shape[1] * self.overlap)
            while idx_end_out <= data.shape[1]:
                # Loop over samples in single trace (inner loop)
                idx_start_inner = 0
                idx_end_inner = self.shape[-1]
                step_inner = int(self.shape[-1] * self.overlap)
                while idx_end_inner <= data.shape[-1]:
                    split_data = data[:, idx_start_out:idx_end_out, idx_start_inner:idx_end_inner]
                    split_target = target[:, idx_start_out:idx_end_out, idx_start_inner:idx_end_inner]

                    # Convert all data to
                    split_data = split_data.astype(np.float32)
                    split_target = split_target.astype(np.float32)

                    # Remove mean from waveform data
                    split_data = split_data - np.mean(split_data)

                    # Normalize split data
                    if self.norm == "peak":
                        split_data = split_data / np.max(np.abs(split_data))
                    elif self.norm == "std":
                        split_data = (split_data - np.mean(split_data)) / np.std(split_data)

                    # Append data and target to self.data
                    self.data.append((split_data,
                                      split_target))

                    # Update inner indices
                    idx_start_inner += step_inner
                    idx_end_inner += step_inner

                # Update outer indices
                idx_start_out += step_out
                idx_end_out += step_out

            # # Create input and target function from data
            # target_p = np.empty(shape=data.shape)
            # for idx in range(data.shape[1]):
            #     # target_p[:, idx, :] = gaussian_pick(onset=metadata.loc[idx, self.metadata_key],
            #     #                                     length=data.shape[2],
            #     #                                     sigma=self.sigma)
            #     # target_p[:, idx, :] = ramp_pick(onset=metadata.loc[idx, self.metadata_key],
            #     #                                 length=data.shape[2],
            #     #                                 slope=0.05)
            #     target_p[:, idx, :] = heavyside(onset=metadata.loc[idx, self.metadata_key],
            #                                     length=data.shape[2])

            # # Split full data into small chunks of size self.shape
            # num_rows = data.shape[1] // self.shape[1]  # Number of traces
            # num_cols = data.shape[2] // self.shape[2]  # Length of each trace
            #
            # for nr in range(num_rows):
            #     for nc in range(num_cols):
            #         split_data = data[
            #                      :,
            #                      int(nr * self.shape[1]):int((nr + 1) * self.shape[1]),
            #                      int(nc * self.shape[2]):int((nc + 1) * self.shape[2])
            #                      ]
            #         split_target_p = target_p[
            #                          :,
            #                          int(nr * self.shape[1]):int((nr + 1) * self.shape[1]),
            #                          int(nc * self.shape[2]):int((nc + 1) * self.shape[2])
            #                            ]
            #
            #         # Convert all data to
            #         split_data = split_data.astype(np.float32)
            #         split_target_p = split_target_p.astype(np.float32)
            #
            #         # Normalize split data
            #         if self.norm == "peak":
            #             split_data = split_data / np.max(np.abs(split_data))
            #         elif self.norm == "std":
            #             split_data = (split_data - np.mean(split_data)) / np.std(split_data)
            #
            #         # Append data and target to self.data
            #         self.data.append((split_data - np.mean(split_data),
            #                           split_target_p))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


class DataSetTest1OutChannel(Dataset):

    def __init__(self,
                 npz_files: list):

        self.data = []
        for filename in npz_files:
            data = np.load(file=filename)["data"]
            self.data.append((data[0, :],   # Waveform data
                              data[1, :]))  # Gaussian probabilites

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def classes(self):
        return self.data.classes


class DataSetTest2OutChannel(Dataset):
    """
    Read traces and probabilites as inout and returns traces, probabilites and noise as output,
    ie shape is
    """
    def __init__(self,
                 npz_files: list):

        self.data = []
        for filename in npz_files:
            data = np.load(file=filename)["data"]
            trace = data[0, :]
            probs = data[1, :]
            noises = 1 - probs
            out = np.empty(shape=(2, *probs.shape[1:]))
            out[0, :] = probs
            out[1, :] = noises

            self.data.append((trace,   # Waveform data
                              out))  # Gaussian probabilites

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def classes(self):
        return self.data.classes


class DataSetTest3OutChannel(Dataset):

    def __init__(self,
                 npz_files: list):

        self.data = []
        for filename in npz_files:
            data = np.load(file=filename)["data"]
            psn = np.array((data[1, 0, :],
                            np.zeros(shape=data[1, 0, :].shape),
                            np.zeros(shape=data[1, 0, :].shape))
                           )

            self.data.append((data[0, :],   # Waveform data
                              psn)          # Gaussian probabilites
                             )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def classes(self):
        return self.data.classes


class DataSet1D(Dataset):

    def __init__(self,
                 npz_files: list):
        self.data = []
        for filename in npz_files:
            data = np.load(file=filename)["data"]
            self.data.append((data[0][0, :],
                              data[1][0, :]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def predict_dataset(data: np.array,
                    model_shape: tuple,
                    sampling_rate,
                    model,
                    overlap: float=0.5,
                    filter_kwargs: Union[None, dict]=None):
    resampled_data = np.empty(shape=(data.shape[0], model_shape[-1]))  # Shape num. of traces times resampled length of each trace

    # Resample each trace
    for idx in tqdm.tqdm(range(data.shape[0])):
        trace = obspy.Trace(data=data[idx, :], header=dict(sampling_rate=sampling_rate,
                                                            component="H"))

        if filter_kwargs:
            trace.filter(**filter_kwargs)

        # Resampling trace to required sample length, i.e. model_shape[-1]
        sampling_rate_factor = data.shape[-1] / model_shape[-1]
        new_sampling_rate = sampling_rate / sampling_rate_factor
        trace.resample(sampling_rate=new_sampling_rate)

        # Write trace to array
        resampled_data[idx, :] = trace.data / np.max(np.abs(trace.data))

    # Reshape resampled data
    resampled_data = np.reshape(resampled_data,
                                newshape=(model_shape[0], data.shape[0], model_shape[-1]))

    # Compute prediction for each chunk in resampled data
    predicted_output = np.zeros(shape=resampled_data.shape)
    idx_trace_start = 0
    idx_trace_end = int(model_shape[1])
    step_trace = model_shape[1] - int(model_shape[1] * overlap)
    while idx_trace_end <= resampled_data.shape[1]:
        with torch.no_grad():  # Predict for batch_size = 1
            batch_data = np.reshape(resampled_data[:, idx_trace_start:idx_trace_end, :],
                                    newshape=(1, *model_shape))
            torch_tensor = torch.Tensor(batch_data)
            prediction = model(torch_tensor)

        # Write prediction into output and stack results
        predicted_output[0, idx_trace_start:idx_trace_end, :] += prediction.detach().numpy()[0, 0, :, :]

        # Update steps
        idx_trace_start += int(step_trace)
        idx_trace_end += int(step_trace)

    # Predict last traces, since last traces might be missing
    with torch.no_grad():
        batch_data = np.reshape(resampled_data[:, -model_shape[1]:, :],
                                newshape=(1, *model_shape))
        torch_tensor = torch.Tensor(batch_data)
        prediction = model(torch_tensor)

    predicted_output[0, -model_shape[1]:, :] += prediction.detach().numpy()[0, 0, :, :]

    # Normalize predicted output to [0, 1]
    for idx in range(predicted_output.shape[1]):
        if np.max(predicted_output[0, idx, :]) > 0:
            predicted_output[0, idx, :] = predicted_output[0, idx, :] / np.max(predicted_output[0, idx, :])

    return predicted_output[0, :]


# def detect_phases(prediction: np.array,
#                   threshold: float=0.75):
#     """

#     :param prediction:
#     :param threshold:
#     :return:
#     """
#     detections_samp = np.empty(prediction.shape[0])
#     for idx in range(prediction.shape[0]):
#         indices = np.where(prediction[idx, :] >= threshold)[0]
#         if len(indices) > 0:
#             detections_samp[idx] = indices[0]
#         else:
#             detections_samp[idx] = np.nan

#     # Check whether pick detections are consistent, i.e., if neighbouring picks are close to each other

#     return detections_samp




def detect_phases(prediction: np.array,
                  sigma=15) -> list:

    edges = feature.canny(prediction,
                          sigma=sigma)
    detections_edges = []
    for j in range(edges.shape[0]):
        det = np.where(edges[j, :]==True)[0]
        if len(det) == 0:
            detections_edges.append(0)
        else:
            detections_edges.append(det[0])

    return np.array(detections_edges)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        weights_init_kaiming(m)
    elif isinstance(m, nn.BatchNorm2d):
        weights_init_kaiming(m)


def normalize_batch(batch, single_trace=True):
    # Normalize each trace to [-1, 1]
    if single_trace is True:
        for batch_id in range(batch[0].shape[0]):
            for k in range(batch[0].shape[2]):
                batch[0][batch_id, 0, k, :] = batch[0][batch_id, 0, k, :] / np.max(np.abs(batch[0][batch_id, 0, k, :].detach().numpy()))
    else:  # Normalize all traces of a common midpoint gather
        for batch_id in range(batch[0].shape[0]):
            batch[0][batch_id, :, :, :] = batch[0][batch_id, :, :, :] / np.max(np.abs(batch[0][batch_id, :, :, :].detach().numpy()))

    return batch


def padding_conv2d_layers(
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
        kernel_size: tuple[int, int],
        stride: tuple[int, int]
                          ):
    padding = [0] * len(input_shape)
    for idx in range(len(input_shape)):
        pad = ((output_shape[idx] - 1) * stride[idx] + kernel_size[idx] - input_shape[idx]) / 2
        padding[idx] = int(pad)

    return tuple(padding)

def padding_transpose_conv2d_layers(
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
        kernel_size: tuple[int, int],
        stride: tuple[int, int]
                          ):
    padding = [0] * len(input_shape)
    for idx in range(len(input_shape)):
        pad = ((input_shape[idx] - 1) * stride[idx] - output_shape[idx] + kernel_size[idx]) / 2
        padding[idx] = int(pad)

    return tuple(padding)


def output_shape_conv2d_layers(input_shape,
                               padding,
                               kernel_size,
                               stride):
    output_shape = [0] * len(input_shape)
    for idx in range(len(input_shape)):
        out = (input_shape[idx] + 2 * padding[idx] - kernel_size[idx]) / stride[idx] + 1
        output_shape[idx] = int(out)

    return tuple(output_shape)


def output_shape_transpose_conv2_layers(input_shape,
                                        padding,
                                        kernel_size,
                                        stride):
    output_shape = [0] * len(input_shape)
    for idx in range(len(input_shape)):
        out = (input_shape[idx] - 1) * stride[idx] - 2 * padding[idx] + kernel_size[idx]
        output_shape[idx] = int(out)

    return tuple(output_shape)


def rms(x):
    """
    Root-mean-square of array x
    :param x:
    :return:
    """
    # Remove mean
    x = x - np.mean(x)
    return np.sqrt(np.sum(x ** 2) / x.shape[0])


def signal_to_noise_ratio(signal, noise, decibel=True):
    """
    SNR in dB
    :param signal:
    :param noise:
    :param decibel:
    :return:
    """
    if len(signal) != len(noise):
        msg = "Length of signal and noise are not equal."
        warnings.warn(msg)

    if decibel is True:
        value = 20 * np.log10(rms(signal) / rms(noise))
        return value
    else:
        return rms(signal) / rms(noise)


def snr(signal, noise, decibel=True):
    """
    Wrapper for signal-to-noise ratio
    """
    return signal_to_noise_ratio(signal=signal, noise=noise, decibel=decibel)


def snr_pick(trace: obspy.Trace,
             picktime: obspy.UTCDateTime,
             window=5,
             **kwargs):
    """
    Computes SNR with a certain time window around a pick
    """
    pick_sample = int((picktime - trace.stats.starttime) * trace.stats.sampling_rate)
    window_len = int(window * trace.stats.sampling_rate)

    if pick_sample - window_len < 0:
        noise_win_begin = 0
    else:
        noise_win_begin = pick_sample - window_len

    return snr(signal=trace.data[pick_sample:pick_sample + window_len],
               noise=trace.data[noise_win_begin:pick_sample],
               **kwargs)


def add_noise(dataset: np.array,
              scale: tuple[float, float]=(0, 1)):
    scale = np.random.uniform(*scale) * np.max(dataset[0, :, :])  # Uniform scale for all traces in dataset
    for trace_idx in range(dataset.shape[1]):
        gaussian_noise = np.random.randn(dataset.shape[2]).astype(dataset.dtype) * scale
        dataset[0, trace_idx, :] = dataset[0, trace_idx, :] + gaussian_noise

        # Normalize noisy dataset
        dataset[0, trace_idx, :] = dataset[0, trace_idx, :] / np.max(np.abs(dataset[0, trace_idx, :]))

    return dataset



if __name__ == "__main__":
    padding = padding_conv2d_layers(input_shape=(32, 2048),
                                    output_shape=(16, 1024),
                                    kernel_size=(2, 2),
                                    stride=(2, 2))
    print(padding)
