import os
import glob
import json
import pathlib
import random

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.offsetbox import AnchoredText

from scipy.ndimage import gaussian_filter
from fbp.src.utils import detect_phases, predict_dataset, residual_histogram, is_nan
from fbp.src.unet import UNet


# Load npz files for testing and define the trained model
npz_files = glob.glob(
    "/scratch/gpi/seis/jheuel/FirstBreakPicking/test_files_v1/C02_C*"
)[:]
# model_filename = "/scratch/gpi/seis/jheuel/FirstBreakPicking/models/model_unetatt_bp_2.5_16.pt"  # Best working model (said Arnaud)
model_filename = "/scratch/gpi/seis/jheuel/FirstBreakPicking/models/performance_tests/fbp_attention_no_noise_fc_v1.pt"
metadata_path = "/scratch/gpi/seis/jheuel/FirstBreakPicking/metadata_v1"
residual = 0.1  # Residual in seconds to compute metrics when testing models

# Define model (must match with trained model) by loading json file
json_fp = os.path.join(
    pathlib.Path(model_filename).parent, f"{pathlib.Path(model_filename).stem}.json"
)
with open(json_fp, "r") as f_json:
    model_args = json.load(f_json)

model = UNet(**model_args)

# Select a random npz file and read metadata to load manually labeled first break picks
filename_npz = random.choice(npz_files)
print(filename_npz)
data = np.load(filename_npz)["data"]
metadata = pd.read_csv(
    os.path.join(metadata_path, f"metadata{Path(filename_npz).stem}.csv")
)

# Load weights of model
model_weights = torch.load(model_filename, map_location=torch.device("cpu"))
model.load_state_dict(model_weights)
model.eval()

# Predict data with loaded model
shape = (model.in_channels, *model.input_shape)
prediction, detections = predict_dataset(
    data=data,
    model=model,
    metadata=metadata,
    overlap=0.95,
    blinding_x=6,
    stacking="avg",
    filter_kwargs=dict(type="bandpass", freqmin=2.5, freqmax=16),
)

# Apply gaussian filter to smooth edges
prediction = gaussian_filter(prediction, sigma=10)

# Detect first break picks on predicted output, i.e. prediction
# Note, detections from predict_dataset are detections on each predicted trace, i.e. with no overlap
detections_single = detect_phases(prediction=prediction, threshold=0.5)

# Define empty arrays for metrics
true_picks = []
true_positives = []  # List with residuals of all true positive picks, i.e. picks that are detected and included in metadata
false_positives = []  # Picks that are false detected, ie no manually picks is close to that pick

# Plot prediction and traces
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax1.set_xlim([0, data.shape[0]])
normalized_data = np.empty(shape=data.shape)
for l in range(data.shape[0]):
    # Plot each trace
    time = (
        np.arange(len(data[l, :])) / metadata.loc[l, "sampling_rate"]
    )  # Convert to seconds
    trace = data[l, :] - np.mean(data[l, :])
    trace = trace / np.max(np.abs(trace))
    normalized_data[l, :] = trace
    ax1.plot(trace + (l + 0.5), time, color="k", linewidth=0.2)

    # Plot P arrival from metadata
    p_arrival_seconds = (
        metadata.loc[l, "trace_P_arrival_sample"] / metadata.loc[l, "sampling_rate"]
    )
    ax1.plot(
        [l, l + 1],
        [p_arrival_seconds, p_arrival_seconds],
        color="tab:orange",
        linewidth=2,
    )
    ax2.plot(
        [l, l + 1],
        [p_arrival_seconds, p_arrival_seconds],
        color="tab:orange",
        linewidth=2,
    )
    if is_nan(p_arrival_seconds) == False:
        true_picks.append(p_arrival_seconds)

    # Plot prediction
    time_prediction = np.linspace(start=0, stop=time[-1], num=len(prediction[l, :]))

    # Plot predicted first break pick
    # Note, if standard deviation from single predicted picks >= 5, the pick is ignored
    detections_seconds = detections_single[l] / (
        metadata.loc[l, "sampling_rate"] * prediction.shape[1] / data.shape[1]
    )
    if np.std(detections[l]) <= 5:
        ax2.plot(
            [l, l + 1], [detections_seconds, detections_seconds], color="r", linewidth=2
        )

        # Compute metrics
        if np.abs(detections_seconds - p_arrival_seconds) <= residual:
            true_positives.append(detections_seconds - p_arrival_seconds)
        elif np.abs(detections_seconds - p_arrival_seconds) > residual:
            false_positives.append(detections_seconds - p_arrival_seconds)

    # Further improvements:
    # If number of neighbouring picks is below a certain threshold, then ignore the predicted picks

# Plot prediction
ax1.pcolormesh(np.arange(data.shape[0]), time_prediction, prediction.T)
ax2.pcolormesh(np.arange(data.shape[0]), time_prediction, prediction.T, alpha=0.9)
ax2.pcolormesh(
    np.arange(data.shape[0]),
    time,
    normalized_data.T**2,
    cmap="Greys_r",
    alpha=0.45,
    norm=colors.LogNorm(vmin=1e-7, vmax=0.01),
)

# Plot pick performance
fig, ax_residual = plt.subplots(nrows=1, ncols=1)
residual_histogram(
    residuals=true_positives, axes=ax_residual, xlim=(-residual, residual)
)
ax_residual.set_title("First-break residual")
ax_residual.set_xlabel("$t_{pred}$ - $t_{true}$ (s)")
ax_residual.set_ylabel("Count")
tpr_text_box = AnchoredText(
    s=f"TPR: {len(true_positives) / len(true_picks):.2f}",
    frameon=False,
    loc="upper left",
    pad=0.5,
)
plt.setp(tpr_text_box.patch, facecolor="white", alpha=0.5)
ax_residual.add_artist(tpr_text_box)
plt.show()
