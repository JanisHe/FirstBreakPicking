import os
import obspy

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from pathlib import Path
from typing import Union

from fbp.src.utils import heavyside, signal_to_noise_ratio, is_nan


class FBPDataset(Dataset):
    """

    :param npz_files:
    :param metadata_path:
    :param shape:
    :param npz_data_key:
    :param metadata_key:
    :param norm:
    :param overlap:
    :param filter_kwargs:
    """
    def __init__(self,
                 npz_files: list,
                 metadata_path: str,
                 shape: tuple = (1, 32, 2048),
                 npz_data_key: str = "data",
                 metadata_key: str = "trace_P_arrival_sample",
                 norm: str = "peak",
                 overlap: float = 0.5,
                 filter_kwargs: Union[None, dict] = None,
                 reduced_velocity: Union[None, float] = None,
                 reduced_sampling_rate: Union[None, float] = None
                 ):
        self.npz_files = npz_files
        self.shape = shape
        self.metadata_path = metadata_path
        self.npz_data_key = npz_data_key
        self.metadata_key = metadata_key
        self.norm = norm
        self.overlap = overlap
        self.filter_kwargs = filter_kwargs
        self.reduced_velocity = reduced_velocity
        self.reduced_sampling_rate = reduced_sampling_rate

        # Determine overlap in samples, depending on output shape
        self.overlap_samples = int(shape[-1] * self.overlap)

        self._read_data()

    def _read_data(self):
        """

        :return:
        """
        self.data = []
        self.snr_chunk = []  # Storing SNR for each single chunk
        self.snr_list = []
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

                if self.filter_kwargs:
                    trace.filter(**self.filter_kwargs)

                # Resampling trace to required sample length, ie self.shape[-1]
                if not self.reduced_velocity:
                    sampling_rate_factor = data.shape[-1] / self.shape[-1]
                    new_sampling_rate = metadata.loc[idx, "sampling_rate"] / sampling_rate_factor
                    trace.resample(sampling_rate=new_sampling_rate)
                    new_onset = np.ceil(metadata.loc[idx, self.metadata_key] / sampling_rate_factor)
                    resampled_data[:, idx, :] = trace.data
                else:  # Cut data when creating training dataset for reduced traveltime
                    trace.resample(sampling_rate=self.reduced_sampling_rate)
                    offset_m = metadata.loc[idx, "distance"]
                    reduced_s = offset_m / self.reduced_velocity
                    zeros = int(self.reduced_sampling_rate * reduced_s)
                    reduced_data = trace.data[zeros:int(zeros + self.shape[-1])]  # Reduce travel time
                    if len(reduced_data) < self.shape[-1]:
                        reduced_data = np.concatenate([reduced_data, np.zeros(int(self.shape[-1] - len(reduced_data)))])
                    resampled_data[:, idx, :] = reduced_data
                    new_onset = np.ceil(
                        metadata.loc[idx, self.metadata_key] * self.reduced_sampling_rate / metadata.loc[
                            idx, "sampling_rate"] - zeros)  # Reduce first-break pick

                # Determine SNR of trace at arrival (new_onset)
                if is_nan(new_onset) == False:
                    snr = signal_to_noise_ratio(signal=resampled_data[0, idx, int(new_onset): int(new_onset + 50)],
                                                noise=resampled_data[0, idx, int(new_onset - 100):int(new_onset - 50)],
                                                decibel=True)
                else:
                    snr = np.nan
                self.snr_list.append(snr)

                # Build target function for each trace
                target[:, idx, :] = heavyside(onset=new_onset,
                                              length=self.shape[-1])

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

                    # Get sub snr list
                    self.snr_chunk.append(self.snr_list[idx_start_out:idx_end_out])

                    # Append data and target to self.data
                    self.data.append((split_data,
                                      split_target))

                    # Update inner indices
                    idx_start_inner += step_inner
                    idx_end_inner += step_inner

                # Update outer indices
                idx_start_out += step_out
                idx_end_out += step_out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes
