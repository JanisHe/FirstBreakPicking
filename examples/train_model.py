import os
import glob
import json
import torch
import pathlib

from torch.utils.data import DataLoader
from torchsummary import summary

from fbp.src.utils import FBP2OutChannels, MeanSquaredError
from fbp.src.unet import UNet


epochs = 250  # Number of epochs for training
batch_size = 32
learning_rate = 0.001
shape = (1, 32, 2048)  # num_channels, width (num traces), height (trace length)
validation_split = 0.8 # Split for training and validation data
loss_fn = MeanSquaredError()  # Loss function
npz_files = glob.glob("/scratch/gpi/seis/HIPER2/delsuc/Seisbench/PhaseNet_DAS_03_04_2025/npz_32_2048_bp2_5_16/*")  # Saved datasets
model_name = "/scratch/gpi/seis/HIPER2/delsuc/Seisbench/PhaseNet_DAS_03_04_2025/models/model_unetatt_bp_2.5_16_drop_focal.pt"  # Model name

# Start training
split = int(len(npz_files) * validation_split)  # Split dataset for training and validation
train_files = npz_files[:split]
val_files = npz_files[split:]

# Create model
model = UNet(depth=5,
             kernel_size=(3, 7),
             stride=(3, 3),
             skip_connections=True,
             out_channels=2,
             filters_root=8,
             output_activation=torch.nn.Softmax(dim=1),
             drop_rate=0.0,
             attention=True)

# If GPU is available, bring model on GPU for faster training
if torch.cuda.is_available() is True:
    device = "cuda"
    model.cuda()
    print("model on cuda")
else:
    device = "cpu"

# Print model
summary(model,
        input_size=shape,
        device=device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)

# Create loader for training and validation
train_dataset = FBP2OutChannels(npz_files=train_files)
val_dataset = FBP2OutChannels(npz_files=val_files)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

# Start train loop
train_loss = []
valid_loss = []
avg_train_loss = []
avg_valid_loss = []

for epoch in range(epochs):
    num_batches = len(train_dataloader)
    model.train()
    for batch_id, batch in enumerate(train_dataloader):
        pred = model(batch[0].to(model.device))
        loss = loss_fn(pred, batch[1].to(model.device))

        # Do backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute loss for each batch
        train_loss.append(loss.item())

        # Validate model
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                pred = model(batch[0].to(model.device))
                valid_loss.append(loss_fn(pred, batch[1].to(model.device)).item())

        # Determine average training and validation loss
        avg_train_loss.append(sum(train_loss) / len(train_loss))
        avg_valid_loss.append(sum(valid_loss) / len(valid_loss))

        if epoch == 0:  # Save model_args as .json file after finishing of first epoch
            json_fp = os.path.join(pathlib.Path(model_name).parent, f"{pathlib.Path(model_name).stem}.json")
            with open(json_fp, "w") as json_fp:
                json.dump(model.get_model_args(), json_fp, indent=4)

        print(f"Epoch {epoch + 1} | train loss: {avg_train_loss[-1]:.5f} | val loss: {avg_valid_loss[-1]:.5f}")

    # Save model after each epoch
    model.eval()
    torch.save(model.state_dict(), model_name)
