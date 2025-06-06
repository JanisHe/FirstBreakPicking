import warnings
from typing import Union

import numpy as np
import obspy
import pandas as pd
import seisbench.models as sbm  # noqa
import torch
import torch.nn as nn
import torchvision
import tqdm
from obspy import UTCDateTime
from scipy import signal
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


def half_gaussian_pick(onset, length, sigma):
    gaussian = gaussian_pick(onset=onset, length=length, sigma=sigma)

    # Cut gaussian at its maximum value
    half_gaussian = gaussian[: np.argmax(gaussian)]
    ones = np.ones(int(length - len(half_gaussian)))

    return np.concatenate([half_gaussian, ones])


def heavyside(onset, length):
    if is_nan(onset) == True:
        return np.zeros(int(length))
    else:
        x0 = np.zeros(int(onset))
        x1 = np.ones(int(length - onset))

        return np.concatenate([x0, x1])


def boxcar(onset, length, width=20):
    if is_nan(onset) == True:
        return np.zeros(int(length))
    else:
        x0 = np.zeros(int(onset - width / 2))
        x1 = np.ones(int(width))
        x2 = np.zeros(int(length - onset - width / 2))

        return np.concatenate([x0, x1, x2])


def ramp_pick(onset, length, slope: float = 1.0):
    if is_nan(onset) == True:
        return np.zeros(length)
    else:
        onset = int(onset)
        x = np.arange(length)
        x0 = np.zeros(onset)
        x_s = x[onset : int(1 / slope + onset)] * slope - onset * slope
        x1 = np.ones(len(x[int(1 / slope + onset) :]))

        return np.concatenate([x0, x_s, x1])


class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(self, input, target):
        return nn.functional.cross_entropy(input=input, target=target)


class BCELoss:
    def __init__(self):
        pass

    def __call__(self, input, target):
        return nn.functional.binary_cross_entropy(input=input, target=target.float())


class DiceLoss:
    """
    https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
    """
    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class MeanSquaredError:
    def __init__(self):
        pass

    def __call__(self, input, target):
        """ """
        mse = (input - target) ** 2
        mse = mse.mean(-1).sum(-1)
        mse = mse.mean()

        return mse


class FocalLoss:
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, input, target):
        val = torchvision.ops.sigmoid_focal_loss(
            inputs=input,
            targets=target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )

        if self.reduction == "none":
            val = val.mean(-1).sum(-1)
            val = val.mean()

        return val


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7,
                 verbose=False,
                 delta=0,
                 path_checkpoint=None,
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path_checkpoint (str, None): Path for the checkpoint to be saved to. If not None chechpoints are saved.
                            Default: None
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path_checkpoint
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')

        if self.path:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.

    https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    """

    def __init__(
            self,
            best_valid_loss=float('inf'),
            model_name: str = "best_model.pth",
            verbose: bool = False,
            trace_func=print
    ):
        self.best_valid_loss = best_valid_loss
        self.model_name = model_name
        self.verbose = verbose
        self.trace_func = trace_func

    def __call__(
            self,
            current_valid_loss,
            epoch,
            model
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            if self.verbose is True:
                self.trace_func(f"Saving best model for epoch {epoch + 1} as {self.model_name}")
            torch.save(obj=model.state_dict(),
                       f=self.model_name)


class DataSetTest1OutChannel(Dataset):
    def __init__(self, npz_files: list):

        self.data = []
        for filename in npz_files:
            data = np.load(file=filename)["data"]
            self.data.append(
                (data[0, :], data[1, :])  # Waveform data
            )  # Gaussian probabilites

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

    def __init__(self, npz_files: list):
        self.data = []
        for filename in npz_files:
            data = np.load(file=filename)["data"]
            trace = data[0, :] - np.mean(data[0, :])  # Normalized seismic data
            probs = data[1, :]  # First break probabilites
            noises = 1 - probs  # Second output channel
            out = np.empty(shape=(2, *probs.shape[1:]))
            out[0, :] = probs
            out[1, :] = noises

            self.data.append((trace, out))  # Waveform data  # Gaussian probabilites

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def classes(self):
        return self.data.classes


def blinding(
    prediction: np.array, blinding_x: int = 4, blinding_y: int = 20
) -> np.array:
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


def model_prediction(
    data: np.array,
    idx_start: int,
    idx_end: int,
    model,
    predicted_output: np.array,
    model_shape: tuple[int, int, int],
    detections: dict,
    blinding_x: int = 4,
    blinding_y: int = 20,
    detection_threshold: float = 0.5,
    stacking: str = "avg",
):
    """
    Predict data in certain range between idx_start and idx_end and also apply blinding and pick phase
    onsets on each predicted output.
    :param data:
    :param idx_start:
    :param idx_end:
    :param model:
    :param predicted_output:
    :param model_shape:
    :param detections:
    :param blinding_x:
    :param blinding_y:
    :param detection_threshold:
    :param stacking:
    :return:
    """
    batch_data = np.reshape(data[:, idx_start:idx_end, :], newshape=(1, *model_shape))
    torch_tensor = torch.Tensor(batch_data)
    prediction = model(torch_tensor)

    # Apply blinding on prediction in x and y direction (both in samples)
    prediction = blinding(
        prediction=prediction.detach().numpy()[0, 0, :, :],
        blinding_x=blinding_x,
        blinding_y=blinding_y,
    )

    # Predict phase onset on single predictions without any overlapping traces
    detected_phase = detect_phases(prediction, threshold=detection_threshold)
    for idx_detect, value in zip(np.arange(idx_start, idx_end), detected_phase):
        if idx_detect not in detections.keys():
            if idx_detect < 0:
                idx_detect = predicted_output.shape[1] + idx_detect
            detections.update({idx_detect: []})
        if is_nan(value) == False:
            detections[idx_detect].append(value)

    # Write prediction into output and stack results
    if stacking == "avg":
        predicted_output[0, idx_start:idx_end, :] = np.nansum(
            a=[predicted_output[0, idx_start:idx_end, :], prediction], axis=0
        )  # prediction
    elif stacking == "max":
        predicted_output[0, idx_start:idx_end, :] = np.nanmax(
            a=[predicted_output[0, idx_start:idx_end, :], prediction], axis=0
        )

    return predicted_output


def predict_dataset(data: np.array,
                    model,
                    metadata: Union[pd.DataFrame, None] = None,
                    sampling_rate: Union[None, float] = None,
                    overlap: float = 0.5,
                    blinding_x: int = 4,
                    blinding_y: int = 20,
                    stacking: str = "avg",
                    detection_threshold: float = 0.5,
                    filter_kwargs: Union[None, dict] = None,
                    reduced_velocity: Union[float, None] = 7000,
                    distances: Union[list, np.array, None] = None,
                    reduced_sampling_rate: Union[float, None] = None) -> (np.array, dict):
    """
    Predicting a full dataset on a previously loaded model.
    If the model was trained on reduced traveltime data, it is necessary to set values for
    'reduced_velocity', 'distances' and 'reduced_sampling_rate'.

    :param data: Array that contains all trace data
    :param model: Loaded pytorch model for prediction
    :param metadata: pd.Dataframe that contains metadata for each trace. Only column with sampling_rate might be
                     required. Note sampling rate is only read once and is assumed to be constant for all traces.
                     However, if metadata is None (which is default), then sampling_rate has to be set in the
                     arguments of this function.
    :param overlap: Overlap between 0 - 1 of neighboring traces
    :param blinding_x: Set traces at the edges of each chunk of data to np.nan to avoid edge effects.
    :param blinding_y: Set begin and end of each single trace to np.nan
    :param stacking: Method how to add up single prediction of overlapping traces either using "max" (maximum)
                     or "avg" (average). Default is avg
    :param detection_threshold: Threshold between 0 - 1 to detect phase onset. Default is 0.5
    :param sampling_rate: Sampling rate in Hz for all traces. Note the sampling rate must be the same for all traces.
                          Only required if metadata is None
    :param filter_kwargs: Dictionary for spectral filtering of data. Default is None. For a detailed description
                          see obspy filter trace
    :param reduced_velocity: When working with reduced velocity, set the velocity in this case. Note the model
                             should also be trained with the same reduced velocity value. Default is 7000 m/s.
    :param distances: List of np.array that contains distances in m of each station to the shot.
                      Only required when working with reduced velocity model.
    :param reduced_sampling_rate: Resampling original data to a new sampling rate when working with reduced
                                  velocity model.

    :return: Numpy array that contains prediction for each trace and detections for each trace.
             Output shape of  prediction is number of traces times time in samples. Note the time depends on the
             sampling_rate, i.e. the original data are resampled to the new sampling rate.
             Detections is a dictionary whose keys are the trace number in prediction and the
             items are the predicted first breaks for each single prediction, i.e. when overlapping is applied,
             each list can have more than one prediction. These single predictions can be used, for example,
             to compute standard deviation of picks and to sort out picks with high uncertainties.
    """
    model_shape = (
        model.in_channels,
        *model.input_shape,
    )  # Estimate input shape of data from loaded model
    resampled_data = np.empty(
        shape=(data.shape[0], model_shape[-1])
    )  # Shape num. of traces times resampled length of each trace
    detections = (
        {}
    )  # key is trace idx and item is list with detections for each single prediction

    # Check whether reduced traveltime is used
    use_reduced_traveltime = False
    if reduced_velocity and distances is not None and reduced_sampling_rate:
        use_reduced_traveltime = True

    if distances is not None:
        if len(distances) != data.shape[0]:
            msg = f"Length of distances ({len(distances)}) and number of traces ({data.shape[0]}) must be equal!"
            raise ValueError(msg)

    # Set sampling rate for traces either from metadata file or from function arguments
    cutoff_lst = []
    # for idx in tqdm.tqdm(range(data.shape[0])):
    for idx in range(data.shape[0]):
        if not sampling_rate:  # Note, sampling rate is only set once from metadata and is then assumend to be constant
            sampling_rate = metadata.loc[idx, "sampling_rate"]
        if metadata is not None:
            starttime = metadata.loc[idx, "trace_start_time"]
        else:
            starttime = UTCDateTime()

        # Write loaded data into obspy Trace
        trace = obspy.Trace(
            data=data[idx, :],
            header=dict(
                sampling_rate=sampling_rate, starttime=starttime, component="Z"
            ),
        )

        if filter_kwargs:
            trace.filter(**filter_kwargs)

        # Resampling trace to required sample length, i.e. model_shape[-1] or reduced_sampling_rate
        if use_reduced_traveltime is True:
            new_sampling_rate = reduced_sampling_rate
        else:
            sampling_rate_factor = data.shape[-1] / model_shape[-1]
            new_sampling_rate = sampling_rate / sampling_rate_factor
        trace.resample(sampling_rate=new_sampling_rate)

        if use_reduced_traveltime:  # Reduce data by traveltime
            reduced_s = distances[idx] / reduced_velocity
            cutoff = int(reduced_sampling_rate * reduced_s)
            trace_data = trace.data[cutoff: int(cutoff + model_shape[-1])]
            # Convert cutoff with correct reduced sampling rate
            cutoff_lst.append(cutoff)
        else:  # Write data from resampled trace
            trace_data = trace.data

        # Demean and min-max normalize data and write to array for prediction
        trace_data -= np.mean(trace_data)
        resampled_data[idx, :] = trace_data / np.max(np.abs(trace_data))

    # Reshape resampled data
    resampled_data = np.reshape(
        resampled_data, newshape=(model_shape[0], data.shape[0], model_shape[-1])
    )

    # Compute prediction for each chunk in resampled data
    predicted_output = np.zeros(shape=resampled_data.shape)
    trace_count = np.zeros(
        shape=resampled_data.shape[1]
    )  # Count for each trace to calculate average
    idx_trace_start = 0
    idx_trace_end = int(model_shape[1])
    step_trace = model_shape[1] - int(model_shape[1] * overlap)
    while idx_trace_end <= resampled_data.shape[1]:
        with torch.no_grad():  # Predict for batch_size = 1
            predicted_output = model_prediction(
                data=resampled_data,
                idx_start=idx_trace_start,
                idx_end=idx_trace_end,
                model=model,
                blinding_x=blinding_x,
                blinding_y=blinding_y,
                detection_threshold=detection_threshold,
                detections=detections,
                stacking=stacking,
                predicted_output=predicted_output,
                model_shape=model_shape,
            )

        # Update trace count
        trace_count[idx_trace_start:idx_trace_end] += 1

        # Update steps
        idx_trace_start += int(step_trace)
        idx_trace_end += int(step_trace)

    # Predict last traces, since last traces might be missing
    with torch.no_grad():
        predicted_output = model_prediction(
            data=resampled_data,
            idx_start=-model_shape[1],
            idx_end=predicted_output.shape[1],
            model=model,
            predicted_output=predicted_output,
            model_shape=model_shape,
            detections=detections,
            blinding_x=blinding_x,
            blinding_y=blinding_y,
            detection_threshold=detection_threshold,
            stacking=stacking,
        )

        # Update trace count
        trace_count[-model_shape[1]:] += 1

    # Build average of predicted output
    if stacking == "avg":
        predicted_output[0] /= trace_count[:, None]

    # Normalize predicted output to [0, 1]
    for idx in range(predicted_output.shape[1]):
        if np.max(predicted_output[0, idx, :]) > 0:
            predicted_output[0, idx, :] = predicted_output[0, idx, :] / np.max(
                predicted_output[0, idx, :]
            )

    # Append zeros before first-break and ones after first-break when using reduced
    if use_reduced_traveltime is True:
        reduced_len = int(data.shape[-1] * reduced_sampling_rate / sampling_rate)
        reduced_predicted_output = np.zeros(shape=(model_shape[0], data.shape[0], reduced_len))
        for idx in range(data.shape[0]):
            # Fill up predicted values to full output array with correct length of original data
            reduced_predicted_output[0, idx, cutoff_lst[idx]:cutoff_lst[idx] + model_shape[-1]] = predicted_output[0, idx,
                                                                                              :]
            # Fill up predicted_output with ones till end
            reduced_predicted_output[0, idx, cutoff_lst[idx] + model_shape[-1] - blinding_y:] = np.ones(
                shape=int(reduced_len - (cutoff_lst[idx] + model_shape[-1] - blinding_y)))

        predicted_output = reduced_predicted_output

    return predicted_output[0, :], detections


def detect_phases(prediction: np.array,
                  threshold: float = 0.75,
                  blinding_x: int = 4):
    """

    :param prediction:
    :param threshold:
    :param blinding_x:
    :return:
    """
    detections_samp = np.empty(prediction.shape[0])
    for idx in range(prediction.shape[0]):
        indices = np.where(prediction[idx, :] >= threshold)[0]
        if len(indices) > 0:
            detections_samp[idx] = indices[0]
        else:
            detections_samp[idx] = np.nan

    # Set picks at edges to np.nan if blinding_x is set
    if blinding_x > 0:
        detections_samp[:blinding_x] = np.nan
        detections_samp[-blinding_x:] = np.nan

    return detections_samp


def normalize_batch(batch, single_trace=True):
    # Normalize each trace to [-1, 1]
    if single_trace is True:
        for batch_id in range(batch[0].shape[0]):
            for k in range(batch[0].shape[2]):
                batch[0][batch_id, 0, k, :] = batch[0][batch_id, 0, k, :] / np.max(
                    np.abs(batch[0][batch_id, 0, k, :].detach().numpy())
                )
    else:  # Normalize all traces of a common midpoint gather
        for batch_id in range(batch[0].shape[0]):
            batch[0][batch_id, :, :, :] = batch[0][batch_id, :, :, :] / np.max(
                np.abs(batch[0][batch_id, :, :, :].detach().numpy())
            )

    return batch


def padding_conv2d_layers(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
):
    padding = [0] * len(input_shape)
    for idx in range(len(input_shape)):
        pad = (
            (output_shape[idx] - 1) * stride[idx] + kernel_size[idx] - input_shape[idx]
        ) / 2
        padding[idx] = int(pad)

    return tuple(padding)


def padding_transpose_conv2d_layers(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
):
    padding = [0] * len(input_shape)
    for idx in range(len(input_shape)):
        pad = (
            (input_shape[idx] - 1) * stride[idx] - output_shape[idx] + kernel_size[idx]
        ) / 2
        padding[idx] = int(pad)

    return tuple(padding)


def output_shape_conv2d_layers(input_shape, padding, kernel_size, stride):
    output_shape = [0] * len(input_shape)
    for idx in range(len(input_shape)):
        out = (input_shape[idx] + 2 * padding[idx] - kernel_size[idx]) / stride[idx] + 1
        output_shape[idx] = int(out)

    return tuple(output_shape)


def output_shape_transpose_conv2_layers(input_shape, padding, kernel_size, stride):
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
    return np.sqrt(np.sum(x**2) / x.shape[0])


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


def snr_pick(trace: obspy.Trace, picktime: obspy.UTCDateTime, window=5, **kwargs):
    """
    Computes SNR with a certain time window around a pick
    """
    pick_sample = int((picktime - trace.stats.starttime) * trace.stats.sampling_rate)
    window_len = int(window * trace.stats.sampling_rate)

    if pick_sample - window_len < 0:
        noise_win_begin = 0
    else:
        noise_win_begin = pick_sample - window_len

    return snr(
        signal=trace.data[pick_sample : pick_sample + window_len],
        noise=trace.data[noise_win_begin:pick_sample],
        **kwargs,
    )


def add_noise(dataset: np.array, scale: tuple[float, float] = (0, 1)):
    scale = np.random.uniform(*scale) * np.max(
        dataset[0, :, :]
    )  # Uniform scale for all traces in dataset
    for trace_idx in range(dataset.shape[1]):
        gaussian_noise = np.random.randn(dataset.shape[2]).astype(dataset.dtype) * scale
        dataset[0, trace_idx, :] = dataset[0, trace_idx, :] + gaussian_noise

        # Normalize noisy dataset
        dataset[0, trace_idx, :] = dataset[0, trace_idx, :] / np.max(
            np.abs(dataset[0, trace_idx, :])
        )

    return dataset


def residual_histogram(residuals, axes, bins=60, xlim=(-100, 100)):

    counts, bins = np.histogram(residuals, bins=bins, range=xlim)
    axes.hist(bins[:-1], bins, weights=counts, edgecolor="b")

    return axes


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def gaussian_kernel(sigma, normalised=True, truncate=4.0, radius=None):
    """
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1."""
    if not radius:
        kernel_size = 2 * np.round(truncate * sigma) + 1
    else:
        kernel_size = 2 * radius + 1

    gaussian1D = signal.gaussian(kernel_size, sigma)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= 2 * np.pi * (sigma**2)
    return gaussian2D


def predict_with_phasenet(
    data: np.array,
    phasenet_model,
    metadata: pd.DataFrame,
    filter_kwargs: Union[None, dict] = None,
    **kwargs,
):
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
            trace = obspy.Trace(
                data=trace_data[i, :],
                header={
                    "sampling_rate": metadata.loc[idx, "sampling_rate"],
                    "channel": f"HH{c}",
                },
            )
            stream.append(trace=trace)

        # Filter stream
        if filter_kwargs:
            stream.filter(**filter_kwargs)

        # Annotate stream
        picks = phasenet_model.classify(stream, batch_size=64, **kwargs).picks

        # Write picks into output array / convert to samples
        trace_picks = []
        for pick in picks:
            trace_picks.append(
                int(
                    (pick.peak_time - stream[0].stats.starttime)
                    * metadata.loc[idx, "sampling_rate"]
                )
            )

        all_picks.append(trace_picks)

    return all_picks
