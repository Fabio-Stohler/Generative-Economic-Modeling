"""Surrogate neural network training, persistence, and evaluation utilities."""

# %%
# Standard library
from pathlib import Path
import copy
import math
import pickle

# Third-party dependencies
import numpy as np
import pandas as pd
import torch
from tqdm import trange


# %% Normalization layer for the neural network
class NormalizeLayer(torch.nn.Module):
    """Normalize inputs to [-1, 1] based on lower/upper bounds."""
    def __init__(self, lower_bound, upper_bound):
        super(NormalizeLayer, self).__init__()

        # Register the lower bound and upper bound as buffers
        self.register_buffer("lower_bound", lower_bound)
        self.register_buffer("upper_bound", upper_bound)

    def forward(self, x):
        return 2 * (x - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1


# %% Scaling layer for the neural network
class ScaleLayer(torch.nn.Module):
    """Rescale outputs from [-1, 1] to the original data range."""
    def __init__(self, lower_bound, upper_bound):
        super(ScaleLayer, self).__init__()

        # Register the lower bound and upper bound as buffers
        self.register_buffer("lower_bound", lower_bound)
        self.register_buffer("upper_bound", upper_bound)

    def forward(self, x):
        # Scale the output x [-1, 1] to the range [lower_bound, upper_bound]
        ub = self.upper_bound
        lb = self.lower_bound

        return torch.where(ub == lb, x, (x + 1) * (ub - lb) / 2 + lb)


# %% Surrogate neural network that maps from x to y
class Surrogate:
    """Feedforward surrogate model with utilities for training, saving, loading, and evaluation."""
    def __init__(self, data=None):
        # Convert dataset to torch tensors if provided.
        if data is not None:
            # Flatten time/batch dimensions so each row is one training sample.
            self.data = self.flatten(copy.deepcopy(data))
            self.N_x = self.data["x"].shape[-1]
            self.N_y = self.data["y"].shape[-1]
            self.network = self.make_network()
        else:
            self.data = None
            self.N_x = None

        # Store training data
        self.data_train = None
        self.data_validation = None
        self.loss_dict = None

        # Accuracy and diagnostics
        self.accuracy = None
        self.cm = None

    def flatten(self, data):
        """Flatten batch/time dims so each sample is 2D (N x features)."""
        for k in data.keys():
            data[k] = data[k].flatten(start_dim=0, end_dim=-2)
        return data

    def shuffle(self, data):
        """Shuffle samples in unison across all tensors in `data`."""
        N_sample = data["x"].shape[0]

        # Randomly shuffle the dataset
        idx = torch.randperm(N_sample)
        for k in data.keys():
            data[k] = data[k][idx, ...]

        return data

    def split_data(self, data, validation_share=0.2, shuffle=True):
        """Split a dataset dict into train/validation folds."""
        N_sample = data["x"].shape[0]
        N_train = int(N_sample * (1 - validation_share))

        if shuffle:
            data = self.shuffle(data)

        data_train = {}
        data_validation = {}
        for k in data.keys():
            data_train[k] = data[k][:N_train, ...]
            data_validation[k] = data[k][N_train:, ...]

        return data_train, data_validation

    def _resolve_path(self, path, name="surrogate"):
        """Return a pickle file path, accepting either a file or directory."""
        p = Path(path)
        if p.suffix == ".pkl":
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{name}.pkl"

    def save(self, path, name="surrogate"):
        """Save the surrogate to disk via pickle. Accepts file or directory path."""
        file_path = self._resolve_path(path, name)
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        """Load a pickled surrogate from disk."""
        # Load the object
        with open(path, "rb") as file:
            obj = pickle.load(file)

        return obj

    def load_attributes(self, path, name="surrogate"):
        """Populate attributes from a saved surrogate object.

        Accepts either a pickle file path or a directory containing `surrogate.pkl`.
        """
        p = Path(path)
        if p.is_dir():
            p = p / f"{name}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Surrogate pickle not found at {p}")

        with open(p, "rb") as f:
            obj = pickle.load(f)

        # Populate attributes
        self.__dict__.update(obj.__dict__)

    def make_network(self, N_inputs=None, N_outputs=None, hidden=64, layers=3, activation=torch.nn.CELU(), normalize_input=True, scale_output=True):
        """Construct the feedforward network and optional scaling layers."""
        if N_inputs is None:
            N_inputs = self.N_x

        if N_outputs is None:
            N_outputs = self.N_y

        layer_list = []

        # Normalization layer
        if normalize_input:
            # Calculate the lower and upper bounds of the inputs from min and max of the dataset
            lb_input = torch.min(self.data["x"], dim=0)[0].unsqueeze(0)
            ub_input = torch.max(self.data["x"], dim=0)[0].unsqueeze(0)

            # Inputs with same lb and ub are not normalized
            for i in range(N_inputs):
                if lb_input[0, i] == ub_input[0, i]:
                    lb_input[0, i] = lb_input[0, i] - 1
                    ub_input[0, i] = ub_input[0, i] + 1

            layer_list.append(NormalizeLayer(lb_input, ub_input))

        # First layer
        layer_list.append(torch.nn.Linear(N_inputs, hidden))
        layer_list.append(activation)

        # Middle layers
        for _ in range(1, layers):
            layer_list.append(torch.nn.Linear(hidden, hidden))
            layer_list.append(activation)

        # Last layer
        layer_list.append(torch.nn.Linear(hidden, N_outputs))

        # Scale output layer
        if scale_output:
            # Calculate the lower and upper bounds of y from min and max of the dataset
            lb_output = torch.min(self.data["y"], dim=0)[0].unsqueeze(0)
            ub_output = torch.max(self.data["y"], dim=0)[0].unsqueeze(0)

            # Outputs with same lb and ub are not scaled
            for i in range(N_outputs):
                if lb_output[0, i] == ub_output[0, i]:
                    lb_output[0, i] = lb_output[0, i] - 1
                    ub_output[0, i] = ub_output[0, i] + 1

            layer_list.append(ScaleLayer(lb_output, ub_output))

        # Build the network
        self.network = torch.nn.Sequential(*layer_list)

    def train(self, batch=64, epochs=10000, device="cpu", lr=1e-3, eta_min=1e-6, validation_share=0.2, print_after=1000, loss_fun="mse", shuffle_data=True, log_loss=False):
        """Train the surrogate network on the provided dataset."""
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 5, gamma=0.1)

        # Loss function
        if loss_fun == "mse":
            loss_function = torch.nn.functional.mse_loss
        elif loss_fun == "l1":
            loss_function = torch.nn.functional.l1_loss
        elif loss_fun == "smooth_l1":
            loss_function = torch.nn.functional.smooth_l1_loss
        elif loss_fun == "huber":
            loss_function = torch.nn.functional.huber_loss
        elif loss_fun == "cauchy":

            def loss_function(y_hat, y):
                return torch.log(1 + (y_hat - y).pow(2)).mean()

        elif loss_fun == "arctan":

            def loss_function(y_hat, y):
                return torch.atan((y_hat - y).pow(2)).mean()

        else:
            loss_function = loss_fun

        # Split dataset into training and validation samples and create dataloaders.
        self.data_train, self.data_validation = self.split_data(self.data, validation_share, shuffle=shuffle_data)
        dataset_train = torch.utils.data.TensorDataset(self.data_train["x"], self.data_train["y"])
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch, shuffle=True)

        if validation_share > 0:
            dataset_validation = torch.utils.data.TensorDataset(self.data_validation["x"], self.data_validation["y"])
            dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch, shuffle=True)

        # Print some statistics about the training and validation data.
        print("Training data:")
        print(f"  - Number of samples: {self.data_train['x'].shape[0]}")
        print(f"  - Number of features: {self.data_train['x'].shape[1]}")
        print(f"  - Number of outputs: {self.data_train['y'].shape[1]}")
        print(f"  - Min: {torch.min(self.data_train['x'], dim=0)[0]}")
        print(f"  - Max: {torch.max(self.data_train['x'], dim=0)[0]}")
        print(f"  - Mean: {torch.mean(self.data_train['x'], dim=0)}")
        print(f"  - Std: {torch.std(self.data_train['x'], dim=0)}")

        print("Validation data:")
        if validation_share > 0:
            print(f"  - Number of samples: {self.data_validation['x'].shape[0]}")
            print(f"  - Number of features: {self.data_validation['x'].shape[1]}")
            print(f"  - Number of outputs: {self.data_validation['y'].shape[1]}")
            print(f"  - Min: {torch.min(self.data_validation['x'], dim=0)[0]}")
            print(f"  - Max: {torch.max(self.data_validation['x'], dim=0)[0]}")
            print(f"  - Mean: {torch.mean(self.data_validation['x'], dim=0)}")
            print(f"  - Std: {torch.std(self.data_validation['x'], dim=0)}")

        # Set the network to train mode
        self.network.train()
        self.network.to(device)

        # Progress bar
        pbar = trange(epochs)

        # Dictionary for loss
        self.loss_dict = {
            "iteration": [],
            "running_loss": [],
            "training_loss": [],
            "validation_loss": [],
        }

        # Training loop over epochs
        for i in pbar:
            running_loss = []
            for x, y in dataloader_train:
                # Move to device
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                y_hat = self.network(x)

                # Loss
                loss = loss_function(y_hat, y)

                # Running loss
                running_loss.append(loss.item())

                # Backward pass
                optimizer.zero_grad()
                if log_loss:
                    loss = torch.log(loss + 1e-4)  # Avoid log(0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()

            # Update learning rate
            scheduler.step()

            # Training loss
            with torch.no_grad():
                training_loss = 0.0
                for x, y in dataloader_train:
                    x = x.to(device)
                    y = y.to(device)
                    y_hat = self.network(x)
                    training_loss += torch.nn.functional.mse_loss(y_hat, y, reduction="sum").item()
                training_loss = training_loss / len(dataset_train)

            # Validation loss
            if validation_share > 0:
                with torch.no_grad():
                    validation_loss = 0.0
                    for x, y in dataloader_validation:
                        x = x.to(device)
                        y = y.to(device)
                        y_hat = self.network(x)
                        validation_loss += torch.nn.functional.mse_loss(y_hat, y, reduction="sum").item()
                    validation_loss = validation_loss / len(dataset_validation)
            else:
                validation_loss = math.nan

            # Average the running loss
            running_loss = torch.mean(torch.tensor(running_loss)).item()

            # Record training and validation loss in the dictionary
            self.loss_dict["iteration"].append(i)
            self.loss_dict["running_loss"].append(running_loss)
            self.loss_dict["training_loss"].append(training_loss)
            self.loss_dict["validation_loss"].append(validation_loss)

            # Print training and validation loss in the progress bar after x epochs
            if (i + 1) % print_after == 0:

                # Update progress bar
                pbar.set_postfix(
                    {
                        "lr:": scheduler.get_last_lr()[0],
                        "loss": running_loss,
                        "training": training_loss,
                        "validation": validation_loss,
                    }
                )

        # Set network to evaluation mode
        self.network.eval()


# %%
