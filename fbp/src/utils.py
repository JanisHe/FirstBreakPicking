import tqdm
import warnings
import torch
import obspy
import torchvision

import numpy as np
import pandas as pd
import torch.nn as nn

from typing import Union
from obspy import UTCDateTime
from scipy import signal
import seisbench.models as sbm  # noqa

from torch.utils.data import Dataset


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


class FocalLoss:
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2,
                 reduction: str = "mean"):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self,
                 input,
                 target):
        val =  torchvision.ops.sigmoid_focal_loss(inputs=input,
                                                  targets=target,
                                                  alpha=self.alpha,
                                                  gamma=self.gamma,
                                                  reduction=self.reduction)

        if self.reduction == "none":
            val = val.mean(-1).sum(-1)
            val = val.mean()

        return val


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


class FBP2OutChannels(Dataset):
    """
    Reads input data as list, loads data and creates two output channels to train the CNN.
    The first output channel is a segmentation map with zeros before the first break pick and ones after and the second
    output channel has ones before the first break pick and zeros after, i.e. the opposite of the first channel.
    :param npz_files:
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


def blinding(prediction: np.array,
             blinding_x: int = 4,
             blinding_y: int = 20) -> np.array:
    """
    Apply blinding on prediction in x and y direction (both in samples)

    :param prediction:
    :param blinding_x:
    :param blinding_y:
    :return:
    """
    if blinding_x > 0:
        prediction[:blinding_x, :] = np.nan
        prediction[-blinding_x:, :] = np.nan
    if blinding_y > 0:
        prediction[:, :blinding_y] = np.nan
        prediction[:, -blinding_y:] = np.nan

    return prediction


def predict_dataset(data: np.array,
                    model,
                    metadata: Union[pd.DataFrame, None] = None,
                    overlap: float=0.5,
                    blinding_x: int=4,
                    blinding_y: int=20,
                    stacking: str = "avg",
                    detection_threshold: float = 0.5,
                    sampling_rate: Union[None, float]=None,
                    filter_kwargs: Union[None, dict]=None):

    model_shape = (model.in_channels, *model.input_shape)  # Estimate input shape of data from loaded model
    resampled_data = np.empty(shape=(data.shape[0], model_shape[-1]))  # Shape num. of traces times resampled length of each trace
    detections = {}  # key is trace idx and item is list with detections for each single prediction

    # Resample each trace
    for idx in tqdm.tqdm(range(data.shape[0])):
        if not sampling_rate:
            sampling_rate = metadata.loc[idx, "sampling_rate"]
        if metadata is not None:
            starttime = metadata.loc[idx, "trace_start_time"]
        else:
            starttime = UTCDateTime()

        trace = obspy.Trace(data=data[idx, :], header=dict(sampling_rate=sampling_rate,
                                                           starttime=starttime,
                                                           component="Z"))

        if filter_kwargs:
            trace.filter(**filter_kwargs)

        # Resampling trace to required sample length, i.e. model_shape[-1]
        sampling_rate_factor = data.shape[-1] / model_shape[-1]
        new_sampling_rate = sampling_rate / sampling_rate_factor
        trace.resample(sampling_rate=new_sampling_rate)

        # Write trace to array
        trace.data -= np.mean(trace.data)
        resampled_data[idx, :] = trace.data / np.max(np.abs(trace.data))

    # Reshape resampled data
    resampled_data = np.reshape(resampled_data,
                                newshape=(model_shape[0], data.shape[0], model_shape[-1]))

    # Compute prediction for each chunk in resampled data
    predicted_output = np.zeros(shape=resampled_data.shape)
    trace_count = np.zeros(shape=resampled_data.shape[1])  # Count for each trace to calculate average
    idx_trace_start = 0
    idx_trace_end = int(model_shape[1])
    step_trace = model_shape[1] - int(model_shape[1] * overlap)
    while idx_trace_end <= resampled_data.shape[1]:
        with torch.no_grad():  # Predict for batch_size = 1
            batch_data = np.reshape(resampled_data[:, idx_trace_start:idx_trace_end, :],
                                    newshape=(1, *model_shape))
            torch_tensor = torch.Tensor(batch_data)
            prediction = model(torch_tensor)
        
        # Apply blinding on prediction in x and y direction (both in samples)
        prediction = blinding(prediction=prediction.detach().numpy()[0, 0, :, :],
                              blinding_x=blinding_x,
                              blinding_y=blinding_y)

        # Predict phase onset on single predictions without any overlapping traces
        detected_phase = detect_phases(prediction,
                                       threshold=detection_threshold)
        for idx_detect, value in zip(np.arange(idx_trace_start, idx_trace_end), detected_phase):
            if idx_detect not in detections.keys():
                detections.update({idx_detect: []})
            if is_nan(value) == False:
                detections[idx_detect].append(value)

        # Write prediction into output and stack results
        if stacking == "avg":
            predicted_output[0, idx_trace_start:idx_trace_end, :] = np.nansum(a=[predicted_output[0,
                                                                                 idx_trace_start:idx_trace_end,
                                                                                 :],
                                                                                 prediction],
                                                                              axis=0)    #prediction
        elif stacking == "max":
            predicted_output[0, idx_trace_start:idx_trace_end, :] = np.nanmax(a=[predicted_output[0,
                                                                                 idx_trace_start:idx_trace_end,
                                                                                 :],
                                                                                 prediction],
                                                                              axis=0)

        # Update trace count
        trace_count[idx_trace_start:idx_trace_end] += 1

        # Update steps
        idx_trace_start += int(step_trace)
        idx_trace_end += int(step_trace)

    # Predict last traces, since last traces might be missing
    with torch.no_grad():
        batch_data = np.reshape(resampled_data[:, -model_shape[1]:, :],
                                newshape=(1, *model_shape))
        torch_tensor = torch.Tensor(batch_data)
        prediction = model(torch_tensor)

        # Apply blinding on prediction in x and y direction (both in samples)
        prediction = blinding(prediction=prediction.detach().numpy()[0, 0, :, :],
                              blinding_x=blinding_x,
                              blinding_y=blinding_y)

        # TODO: This code is doubled
        detected_phase = detect_phases(prediction,
                                       threshold=detection_threshold)
        for idx_detect, value in zip(np.arange(data.shape[0]-model_shape[1], data.shape[0]), detected_phase):
            if idx_detect not in detections.keys():
                detections.update({idx_detect: []})
            if is_nan(value) == False:
                detections[idx_detect].append(value)

        if stacking == "avg":
            predicted_output[0, -model_shape[1]:, :] = np.nansum(a=[predicted_output[0, -model_shape[1]:, :],
                                                                    prediction],
                                                                 axis=0)
        elif stacking == "max":
            predicted_output[0, -model_shape[1]:, :] = np.nanmax(a=[predicted_output[0, -model_shape[1]:, :],
                                                                    prediction],
                                                                 axis=0)
        trace_count[-model_shape[1]:] += 1

    # Build average of predicted output
    if stacking == "avg":
        predicted_output[0] /= trace_count[:, None]

    # Normalize predicted output to [0, 1]
    for idx in range(predicted_output.shape[1]):
        if np.max(predicted_output[0, idx, :]) > 0:
            predicted_output[0, idx, :] = predicted_output[0, idx, :] / np.max(predicted_output[0, idx, :])

    return predicted_output[0, :], detections


def detect_phases(prediction: np.array,
                  threshold: float=0.75):
    """

    :param prediction:
    :param threshold:
    :return:
    """
    detections_samp = np.empty(prediction.shape[0])
    for idx in range(prediction.shape[0]):
        indices = np.where(prediction[idx, :] >= threshold)[0]
        if len(indices) > 0:
            detections_samp[idx] = indices[0]
        else:
            detections_samp[idx] = np.nan

    # Check whether pick detections are consistent, i.e., if neighbouring picks are close to each other

    return detections_samp


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


def residual_histogram(residuals,
                       axes,
                       bins=60,
                       xlim=(-100, 100)):

    counts, bins = np.histogram(residuals,
                                bins=bins,
                                range=xlim)
    axes.hist(bins[:-1], bins,
              weights=counts,
              edgecolor="b")

    return axes


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian_kernel(sigma, normalised=True, truncate=4.0, radius=None):
    '''
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    if not radius:
        kernel_size = 2 * np.round(truncate * sigma) + 1
    else:
        kernel_size = 2 * radius + 1

    gaussian1D = signal.gaussian(kernel_size, sigma)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2 * np.pi * (sigma ** 2))
    return gaussian2D


def predict_with_phasenet(data: np.array,
                          phasenet_model,
                          metadata: pd.DataFrame,
                          filter_kwargs: Union[None, dict] = None,
                          **kwargs):
    """

    :param data:
    :param phasenet_model:
    :param metadata:
    :param kwargs:
    :return:
    """
    all_picks = []
    for idx in range(data.shape[0]):
        zeros = np.zeros(len(data[idx, :]))
        trace_data = np.array([data[idx, :], zeros, zeros])
        stream = obspy.Stream()
        for i, c in zip(range(3), "ZNE"):
            trace = obspy.Trace(data=trace_data[i, :],
                                header={"sampling_rate": metadata.loc[idx, "sampling_rate"],
                                        "channel": f"HH{c}"})
            stream.append(trace=trace)

        # Filter stream
        if filter_kwargs:
            stream.filter(**filter_kwargs)

        # Annotate stream
        picks = phasenet_model.classify(stream, batch_size=64, **kwargs).picks

        # Write picks into output array / convert to samples
        trace_picks = []
        for pick in picks:
            trace_picks.append(int((pick.peak_time - stream[0].stats.starttime) * metadata.loc[idx, "sampling_rate"]))

        all_picks.append(trace_picks)

    return all_picks
