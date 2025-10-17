"""
Find best model by optimizing the models with propulate and testing
each model on the test dataset.
"""

import os
import glob
import json
import torch
import time
import logging
import random
import seisbench  # noqa
import socket
import pathlib

import numpy as np
import pandas as pd

import datetime as dt
from mpi4py import MPI
import torch.distributed as dist
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from propulate import Islands
from propulate.utils import get_default_propagator, set_logger_config
from typing import Union

from fbp.src.utils import (
    FBP2OutChannels,
    EarlyStopping,
    SaveBestModel,
    predict_dataset,
    detect_phases,
    is_nan,
    MeanSquaredError,
    BCELoss,
    DiceLoss,
    FocalLoss,
)
from fbp.src.unet import UNet


log = logging.getLogger("propulate")  # Get logger instance.
SUBGROUP_COMM_METHOD = "nccl-slurm"
GPUS_PER_NODE = 4


def torch_process_group_init_propulate(
    subgroup_comm: MPI.Comm, method: str, trace_func=print
) -> None:
    """
    Create the torch process group of each multi-rank worker from a subgroup of the MPI world.

    Parameters
    ----------
    subgroup_comm : MPI.Comm
        The split communicator for the multi-rank worker's subgroup. This is provided to the individual's loss function
        by the ``Islands`` class if there are multiple ranks per worker.
    method : str
        The method to use to initialize the process group.
        Options: [``nccl-slurm``, ``nccl-openmpi``, ``gloo``]
        If CUDA is not available, ``gloo`` is automatically chosen for the method.
    trace_func: prints output. Default is print statement
    """
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_ROOT

    comm_rank, comm_size = subgroup_comm.rank, subgroup_comm.size

    # Get master address and port
    # Don't want different groups to use the same port.
    subgroup_id = MPI.COMM_WORLD.rank // comm_size
    port = 29500 + subgroup_id

    if comm_size == 1:
        return
    master_address = f"{socket.gethostname()[:-7]}i"  # THIS IS THE NEW BIT! IT WILL PULL OUT THE rank-0 NODE NAME
    # master_address = f"{socket.gethostname()}"
    # Each multi-rank worker rank needs to get the hostname of rank 0 of its subgroup.
    master_address = subgroup_comm.bcast(str(master_address), root=0)

    # Save environment variables.
    os.environ["MASTER_ADDR"] = master_address
    # Use the default PyTorch port.
    os.environ["MASTER_PORT"] = str(port)

    if not torch.cuda.is_available():
        method = "gloo"
        trace_func("No CUDA devices found: Falling back to gloo.")
    else:
        trace_func(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        num_cuda_devices = torch.cuda.device_count()
        device_number = MPI.COMM_WORLD.rank % num_cuda_devices
        trace_func(f"device count: {num_cuda_devices}, device number: {device_number}")
        torch.cuda.set_device(device_number)

    time.sleep(0.001 * comm_rank)  # Avoid DDOS'ing rank 0.
    if method == "nccl-openmpi":  # Use NCCL with OpenMPI.
        dist.init_process_group(
            backend="nccl",
            rank=comm_rank,
            world_size=comm_size,
        )

    elif method == "nccl-slurm":  # Use NCCL with a TCP store.
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=comm_size,
            is_master=(comm_rank == 0),
            timeout=dt.timedelta(seconds=900),
        )
        dist.init_process_group(
            backend="nccl",
            store=wireup_store,
            world_size=comm_size,
            rank=comm_rank,
        )
    elif method == "gloo":  # Use gloo.
        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=comm_size,
            is_master=(comm_rank == 0),
            timeout=dt.timedelta(seconds=900),
        )
        dist.init_process_group(
            backend="gloo",
            store=wireup_store,
            world_size=comm_size,
            rank=comm_rank,
        )
    else:
        raise NotImplementedError(
            f"Given 'method' ({method}) not in [nccl-openmpi, nccl-slurm, gloo]!"
        )

    # Call a barrier here in order for sharp to use the default comm.
    if dist.is_initialized():
        dist.barrier()
        disttest = torch.ones(1)
        if method != "gloo":
            disttest = disttest.cuda()

        dist.all_reduce(disttest)
        assert disttest[0] == comm_size, "Failed test of dist!"
    else:
        disttest = None
    trace_func(
        f"Finish subgroup torch.dist init: world size: {dist.get_world_size()}, rank: {dist.get_rank()}"
    )


def get_data_loaders(train_files: list, val_files: list, batch_size: int):
    # Create loader for training and validation
    train_dataset = FBP2OutChannels(npz_files=train_files)
    val_dataset = FBP2OutChannels(npz_files=val_files)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=6,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=6,
    )

    return train_dataloader, val_dataloader


def trainer(
    epochs: int, train_dataloader, val_dataloader, model, loss_fn, optimizer, model_name
):
    early_stopping = EarlyStopping(patience=20, verbose=False, path_checkpoint=None)
    best_model = SaveBestModel(model_name=model_name)

    # Start train loop
    train_loss = []
    valid_loss = []
    avg_train_loss = []
    avg_valid_loss = []

    for epoch in range(epochs):
        for batch_id, batch in enumerate(train_dataloader):
            try:  # If model does not work correct
                pred = model(batch[0].to(model.device))
            except RuntimeError:
                raise RuntimeError
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
        best_model(current_valid_loss=avg_valid_loss[-1], epoch=epoch, model=model)

        if epoch == 0:  # Save model_args as .json file after finishing of first epoch
            json_fp = os.path.join(
                pathlib.Path(model_name).parent, f"{pathlib.Path(model_name).stem}.json"
            )
            with open(json_fp, "w") as json_fp:
                json.dump(model.get_model_args(), json_fp, indent=4)

        # print(f"Epoch {epoch + 1} | train loss: {avg_train_loss[-1]:.5f} | val loss: {avg_valid_loss[-1]:.5f}",
        #       flush=True)

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

    return


def model_tester(
    model_name: str,
    metadata_path: str,
    test_files: list,
    reduced_traveltime: bool,
    std_threshold: float = 5,
    residual: float = 0.05,
):
    # Define model (must match with trained model) by loading json file
    json_fp = os.path.join(
        pathlib.Path(model_name).parent, f"{pathlib.Path(model_name).stem}.json"
    )
    with open(json_fp, "r") as f_json:
        model_args = json.load(f_json)

    # Load model
    model = UNet(**model_args)
    model_weights = torch.load(model_name, map_location=torch.device("cpu"))
    model.load_state_dict(model_weights)
    model.eval()

    # Define empty arrays for metrics
    true_picks = []
    true_positives = []  # List with residuals of all true positive picks, i.e. picks that are detected and included in metadata
    false_positives = []  # Picks that are false detected, ie no manually picks is close to that pick

    # Set values for reduced traveltime
    if reduced_traveltime is True:
        reduced_velocity = 7000
        reduced_sampling_rate = 75
    else:
        reduced_velocity = None
        reduced_sampling_rate = None

    # Loop over each testfile and predict phase onsets
    for filename in test_files:
        data = np.load(filename)["data"]
        metadata = pd.read_csv(
            os.path.join(metadata_path, f"metadata{pathlib.Path(filename).stem}.csv")
        )
        prediction, detections = predict_dataset(
            data=data,
            model=model,
            metadata=metadata,
            overlap=0.95,
            blinding_x=6,
            stacking="avg",
            reduced_velocity=reduced_velocity,
            distances=metadata["distance"],
            reduced_sampling_rate=reduced_sampling_rate,
            filter_kwargs=dict(type="bandpass", freqmin=2.5, freqmax=16),
        )

        # Apply gaussian filter to smooth edges
        prediction = gaussian_filter(prediction, sigma=10)

        # Detect first break picks on predicted output, i.e. prediction
        # Note, detections from predict_dataset are detections on each predicted trace, i.e. with no overlap
        detections_single = detect_phases(prediction=prediction, threshold=0.5)

        # Loop over each single trace
        for l_idx in range(data.shape[0]):
            p_arrival_seconds = (
                metadata.loc[l_idx, "trace_P_arrival_sample"]
                / metadata.loc[l_idx, "sampling_rate"]
            )
            if not is_nan(p_arrival_seconds):
                true_picks.append(p_arrival_seconds)

            detections_seconds = detections_single[l_idx] / (
                metadata.loc[l_idx, "sampling_rate"]
                * prediction.shape[1]
                / data.shape[1]
            )
            if np.std(detections[l_idx]) <= std_threshold:
                # Compute metrics
                if np.abs(detections_seconds - p_arrival_seconds) <= residual:
                    true_positives.append(detections_seconds - p_arrival_seconds)
                elif np.abs(detections_seconds - p_arrival_seconds) > residual:
                    false_positives.append(detections_seconds - p_arrival_seconds)

    # Estimate true positive rate and return
    return len(true_positives) / len(true_picks)


def ind_loss(params: dict[str, Union[int, float, str]]) -> float:
    """
    Loss function for evolutionary optimization with Propulate. Minimize the model's negative validation accuracy.

    Parameters
    ----------
    params : Dict[str, int | float | str]
        The hyperparameters to be optimized evolutionarily.

    Returns
    -------
    float
        The trained model's negative validation accuracy.
    """
    if params["skip_connections"] == "True":
        skip_connections = True
    elif params["skip_connections"] == "False":
        skip_connections = False

    if params["attention"] == "True":
        attention = True
    elif params["attention"] == "False":
        attention = False

    model = UNet(
        depth=params["depth"],
        # kernel_size=params["kernel_size"],
        # stride=params["stride"],
        kernel_size=(3, 7),
        stride=(3, 3),
        skip_connections=skip_connections,
        out_channels=2,
        filters_root=params["filters_root"],
        output_activation=torch.nn.Softmax(dim=1),
        drop_rate=params["drop_rate"],
        attention=attention,
    )

    if params["noise_files"] == "True":
        train_files = glob.glob(
            "/scratch/gpi/seis/jheuel/FirstBreakPicking/npz_reduced_v1/*"
        )
        noisy_files = glob.glob(
            "/scratch/gpi/seis/jheuel/FirstBreakPicking/npz_reduced_noise_v1/*"
        )
    else:
        train_files = glob.glob(
            "/scratch/gpi/seis/jheuel/FirstBreakPicking/npz_reduced_v1/*"
        )

    split = int(len(train_files) * 0.8)  # XXX Hard coded validation split of 0.8
    train_files, val_files = train_files[:split], train_files[split:]

    # Append noisy files
    if params["noise_files"] is True:
        split = int(len(noisy_files) * 0.8)
        train_files += noisy_files[:split]
        val_files += noisy_files[split:]

    # Get dataloaders
    train_loader, validation_loader = get_data_loaders(
        train_files, val_files, params["batch_size"]
    )

    # Move model to GPU if GPU is available
    if torch.cuda.is_available():
        model.cuda()

    # Define loss function
    loss_fn = params["loss_fn"]
    if loss_fn == "MeanSquaredError":
        loss_fn = MeanSquaredError()
    elif loss_fn == "DiceLoss":
        loss_fn = DiceLoss()
    elif loss_fn == "FocalLoss":
        loss_fn = FocalLoss()
    elif loss_fn == "BCELoss":
        loss_fn = BCELoss()

    # specify learning rate and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Modify model name for a temporary name to save best model and json file
    model_path, model_name = (
        pathlib.Path(params["model_name"]).parent,
        pathlib.Path(params["model_name"]).stem,
    )
    random_name = int(time.time()) + random.randint(0, int(time.time()))
    model_name_tmp = os.path.join(model_path, f"{model_name}_{random_name}.pt")
    json_tmp = os.path.join(
        pathlib.Path(model_name_tmp).parent, f"{pathlib.Path(model_name_tmp).stem}.json"
    )

    # Train model
    try:  # RuntimeError is raised when model does not work with given parameters
        trainer(
            epochs=params["epochs"],
            train_dataloader=train_loader,
            val_dataloader=validation_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            model_name=model_name_tmp,
        )
    except RuntimeError:
        return 1000

    # Test model on test dataset
    if params["reduced_traveltime"] == "False":
        reduced_traveltime = False
    elif params["reduced_traveltime"] == "True":
        reduced_traveltime = True

    tpr = model_tester(
        model_name=model_name_tmp,
        metadata_path=params["metadata"],
        test_files=glob.glob(params["test_files"]),
        reduced_traveltime=reduced_traveltime,
        std_threshold=params["std_threshold"],
    )

    # Rename json and model wrt to true positive rate (TPR)
    tpr_model_name = os.path.join(model_path, f"{model_name}_{tpr}.pt")
    tpr_json = os.path.join(
        pathlib.Path(tpr_model_name).parent, f"{pathlib.Path(tpr_model_name).stem}.json"
    )
    os.rename(src=model_name_tmp, dst=tpr_model_name)
    os.rename(src=json_tmp, dst=tpr_json)

    # Return 1 - TPR for optimization
    return 1 - tpr


def check_propulate_limits(params: dict) -> dict:
    """
    Check whether one parameter in dictionary params has only a length of one.
    If yes, the same value is appended to the tuple. If only one parameter is in params,
    this parameters is not modified by propulate to find the best hyperparameters.

    This function is necessary to run propulate sucessfull.
    """
    for key, value in params.items():
        if isinstance(value, tuple):
            if len(value) == 1:
                params[key] = tuple([value[0], value[0]])
        elif isinstance(value, list):
            if len(value) == 1:
                params[key] = tuple([value[0], value[0]])
            else:
                params[key] = tuple(value)
        else:
            params[key] = tuple([value, value])

    return params


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    comm.Barrier()
    pop_size = 2 * comm.size  # Breeding population size

    # Set up propulator
    generations = 10
    checkpoint_path = "./propulate_ckpt"
    num_islands = 1
    migration_probability = 0.9
    pollination = True
    ranks_per_worker = 1

    params = {
        "depth": (2, 6),
        "skip_connections": ("True", "False"),
        # "kernel_size": ((3, 7), (3, 7)),
        # "strid": ((3, 3), (3, 3)),
        "filters_root": (2, 4, 8, 16),
        "drop_rate": (0.0, 0.5),
        "attention": ("True", "False"),
        "noise_files": ("True", "False"),
        "test_files": ("/scratch/gpi/seis/jheuel/FirstBreakPicking/test_files_v1/*"),
        "batch_size": (16, 32, 64, 128, 256),
        "loss_fn": ("MeanSquaredError", "DiceLoss", "FocalLoss", "BCELoss"),
        "learning_rate": (0.001, 0.01, 0.1),
        "model_name": (os.path.join(checkpoint_path, "deep_fb_propulate.pt")),
        "epochs": (500),
        "metadata": ("/scratch/gpi/seis/jheuel/FirstBreakPicking/metadata_v1"),
        "reduced_traveltime": ("True", "False"),
        "std_threshold": (0, 100),
    }

    # Check limits for propulate
    limits_dict = check_propulate_limits(params=params)

    rng = random.Random(
        comm.rank
    )  # Set up separate random number generator for evolutionary optimizer.

    # Set up separate logger for Propulate optimization.
    set_logger_config(
        level=logging.INFO,  # Logging level
        log_file="logs/deep_fb_propulate.log",  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    propagator = get_default_propagator(  # Get default evolutionary operator.
        pop_size=pop_size,  # Breeding population size
        limits=limits_dict,  # Search space
        crossover_prob=0.7,  # Crossover probability
        mutation_prob=0.4,  # Mutation probability
        random_init_prob=0.1,  # Random-initialization probability
        rng=rng,  # Separate random number generator for Propulate optimization
    )

    # Set up island model.
    islands = Islands(
        loss_fn=ind_loss,  # Loss function to be minimized
        propagator=propagator,  # Propagator, i.e., evolutionary operator to be used
        rng=rng,  # Separate random number generator for Propulate optimization
        generations=generations,  # Overall number of generations
        num_islands=num_islands,  # Number of islands
        migration_probability=migration_probability,  # Migration probability
        pollination=pollination,  # Whether to use pollination or migration
        checkpoint_path=checkpoint_path,  # Checkpoint path
        # ----- SPECIFIC FOR MULTI-RANK UCS -----
        ranks_per_worker=ranks_per_worker,  # GPUS_PER_NODE,  # Number of ranks per (multi rank) worker
    )

    # Run actual optimization.
    islands.propulate(
        logging_interval=1,  # Logging interval
        debug=1,  # Debug level
    )
    islands.summarize(
        top_n=5,  # Print top-n best individuals on each island in summary.
        debug=1,  # Debug level
    )
