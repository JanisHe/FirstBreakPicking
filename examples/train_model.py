import os
import glob
import json
import torch
import pathlib

from torch.utils.data import DataLoader
from torchsummary import summary

from fbp.src.utils import FBP2OutChannels, MeanSquaredError, BCELoss, DiceLoss, EarlyStopping, SaveBestModel
from fbp.src.unet import UNet


epochs = 250  # Number of epochs for training
batch_size = 32
learning_rate = 0.001
shape = (1, 32, 2048)  # num_channels, width (num traces), height (trace length)
validation_split = 0.8 # Split for training and validation data
early_stopping = EarlyStopping(patience=20, verbose=False, path_checkpoint=None)  # Initialize early stopping class
# loss_fn = MeanSquaredError()  # Loss function
loss_fn = DiceLoss()
npz_files = glob.glob("/scratch/gpi/seis/jheuel/FirstBreakPicking/npz_v1/*")  # Saved datasets
model_name = "/scratch/gpi/seis/jheuel/FirstBreakPicking/models/fbp_attention_bce_v1.pt"  # Model name

# Start training
split = int(len(npz_files) * validation_split)  # Split dataset for training and validation
train_files = npz_files[:split]
val_files = npz_files[split:]

# Initialize best model
best_model = SaveBestModel(model_name=model_name)

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
    for batch_id, batch in enumerate(train_dataloader):
        pred = model(batch[0].to(model.device))
        loss = loss_fn(pred, batch[1].to(model.device))

        # Do backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute loss for each batch
        train_loss.append(loss.item())

    # Validate model for each epoch
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            pred = model(batch[0].to(model.device))
            valid_loss.append(loss_fn(pred, batch[1].to(model.device)).item())

    # Determine average training and validation loss
    avg_train_loss.append(sum(train_loss) / len(train_loss))
    avg_valid_loss.append(sum(valid_loss) / len(valid_loss))

    # Save model if validation loss decreased
    best_model(current_valid_loss=avg_valid_loss[-1],
               epoch=epoch,
               model=model)

    if epoch == 0:  # Save model_args as .json file after finishing of first epoch
        json_fp = os.path.join(pathlib.Path(model_name).parent, f"{pathlib.Path(model_name).stem}.json")
        with open(json_fp, "w") as json_fp:
            json.dump(model.get_model_args(), json_fp, indent=4)

    print(f"Epoch {epoch + 1} | train loss: {avg_train_loss[-1]:.5f} | val loss: {avg_valid_loss[-1]:.5f}",
          flush=True)

    # Re-open model for next epoch
    model.train()

    # Clear training and validation loss lists for next epoch
    train_loss = []
    valid_loss = []

    # early_stopping needs the validation loss to check if it has decresed,
    # and if it has, it will make a checkpoint of the current model
    early_stopping(avg_valid_loss[-1], model)

    if early_stopping.early_stop:
        print("Validation loss does not decrease further. Early stopping")
        break
