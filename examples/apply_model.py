import glob
import random

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from fbp.src.utils import detect_phases, predict_dataset
from fbp.src.unet import UNet


# Load npz files for testing and define the trained model
npz_files = glob.glob("/scratch/gpi/seis/HIPER2/delsuc/Seisbench/PhaseNet_DAS_03_04_2025/data_npz/*")[:]
model_filename = "/scratch/gpi/seis/HIPER2/delsuc/Seisbench/PhaseNet_DAS_03_04_2025/models/test.pt"

# Define model (must match with trained model)
# Improvement to store arguments in json file and load a separate json file instead
model = UNet(depth=5,
             kernel_size=(3, 7),
             stride=(3, 3),
             skip_connections=True,
             out_channels=2,
             filters_root=8,
             output_activation=torch.nn.Softmax(dim=1),
             drop_rate=0.0,
             attention=True)

# Select a random npz file and read metadata to load manually labeled first break picks
filename_npz = random.choice(npz_files)
print(filename_npz)
data = np.load(filename_npz)["data"]
metadata = pd.read_csv(f"/scratch/gpi/seis/HIPER2/delsuc/Seisbench/PhaseNet_DAS_03_04_2025/metadata/metadata{Path(filename_npz).stem}.csv")

# Load weights of model
model_weights = torch.load(model_filename, map_location=torch.device("cpu"))
model.load_state_dict(model_weights)
model.eval()

# Predict data with loaded model
shape = (model.in_channels, *model.input_shape)
prediction, detections = predict_dataset(data=data,
                                         model_shape=shape,
                                         model=model,
                                         metadata=metadata,
                                         overlap=0.95,
                                         blinding_x=6,
                                         stacking="avg",
                                         filter_kwargs=dict(type="bandpass",
                                                            freqmin=2.5,
                                                            freqmax=16))

# Apply gaussian filter to smooth edges
prediction = gaussian_filter(prediction, sigma=10)

# Detect first break picks on predicted output, i.e. prediction
# Note, detections from predict_dataset are detections on each predicted trace, i.e. with no overlap
detections_single = detect_phases(prediction=prediction,
                                  threshold=0.5)

# Plot prediction and traces
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax1.set_xlim([0, data.shape[0]])
for l in range(data.shape[0]):
    # Plot each trace
    time = np.arange(len(data[l, :])) / metadata.loc[l, "sampling_rate"]  # Convert to seconds
    trace = data[l, :]
    trace = trace / np.max(np.abs(trace))
    ax1.plot(trace + (l + 0.5), time, color="k", linewidth=0.2)

    # Plot P arrival from metadata
    p_arrival_seconds = metadata.loc[l, "trace_P_arrival_sample"] / metadata.loc[l, "sampling_rate"]
    ax1.plot([l, l + 1], [p_arrival_seconds, p_arrival_seconds], color="tab:orange", linewidth=2)
    ax2.plot([l, l + 1], [p_arrival_seconds, p_arrival_seconds], color="tab:orange", linewidth=2)

    # Plot prediction
    time_prediction = np.linspace(start=0,
                                  stop=time[-1],
                                  num=len(prediction[l, :]))

    # Plot predicted first break pick
    # Note, if standard deviation from single predicted picks >= 5, the pick is ignored
    detections_seconds = detections_single[l] / (metadata.loc[l, "sampling_rate"] * prediction.shape[1] / data.shape[1])
    if np.std(detections[l]) <= 5:
        ax2.plot([l, l + 1], [detections_seconds, detections_seconds], color="r", linewidth=2)

    # Further improvements:
    # If number of neighbouring picks is below a certain threshold, then ignore the predicted picks

# Plot prediction
ax1.pcolormesh(np.arange(data.shape[0]), time_prediction, prediction.T)
ax2.pcolormesh(np.arange(data.shape[0]), time_prediction, prediction.T)
plt.show()
