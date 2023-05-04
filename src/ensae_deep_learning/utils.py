import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tf
import tqdm.notebook as tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.utils import make_grid

from ensae_deep_learning.sde import loss_fn


def rotate_90(img):
    return tf.rotate(img, -90)


def hflip(img):
    return tf.hflip(img)


def summary_model(model, config):
    input_data = torch.randn(
        config["batch_size"],
        config["in_channels"],
        config["image_size"][0],
        config["image_size"][1],
    )
    input_t = torch.rand(config["batch_size"])

    summary(model, input_data=[input_data, input_t])


def print_sde_dim(config):
    kernel = 3
    stride = 2

    dim = config["image_size"][0]
    print(dim)

    def hout_down(hin, kernel, stride):
        return int((hin - kernel) / stride + 1)

    dim = hout_down(dim, kernel, 1)
    print(dim)

    dim = hout_down(dim, kernel, stride)
    print(dim)

    dim = hout_down(dim, kernel, stride)
    print(dim)

    dim = hout_down(dim, kernel, stride)
    print(dim)

    def hout_up(hin, kernel, stride):
        return (hin - 1) * stride + (kernel - 1) + 1

    print(dim)

    dim = hout_up(dim, kernel, stride)
    print(dim)

    dim = hout_up(dim, kernel, stride)
    print(dim)

    dim = hout_up(dim, kernel, stride)
    print(dim)

    dim = hout_up(dim, kernel, 1)
    print(dim)


def plot_dataset(dataset, save_dir, filename, unnormalize=None):
    image_list = []
    random.seed(42)
    for _ in range(36):
        index = random.randint(0, len(dataset) - 1)
        image = dataset[index][0]
        if unnormalize:
            image = unnormalize(image)
        image_list.append(image)
    sample_grid = make_grid(image_list, nrow=6)

    plt.figure(figsize=(6, 6))
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(Path(save_dir, filename))
    plt.show()


def run_training(
    score_model,
    dataset_transformed,
    config,
    marginal_prob_std_fn,
    save_dir,
    device,
):
    loss_history_path = Path(save_dir, "loss_history.pickle")

    if not loss_history_path.exists():
        data_loader = DataLoader(
            dataset_transformed,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=6,
        )

        optimizer = Adam(score_model.parameters(), lr=config["lr"])
        tqdm_epoch = tqdm.trange(config["n_epochs"])

        loss_history = []

        for _epoch in tqdm_epoch:
            avg_loss = 0.0
            num_items = 0
            for x, _y in data_loader:
                x = x.to(device)
                loss = loss_fn(score_model, x, marginal_prob_std_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
            # Print the averaged training loss so far.
            tqdm_epoch.set_description(f"Average Loss: {avg_loss / num_items:5f}")
            # Update the checkpoint after each epoch of training.
            torch.save(score_model.state_dict(), Path(save_dir, "model.pth"))

            loss_history.append(avg_loss / num_items)

        with open(loss_history_path, "wb") as f:
            pickle.dump(loss_history, f)
    else:
        print("Loading previous training...")
        with open(loss_history_path, "rb") as f:
            loss_history = pickle.load(f)
    return loss_history


def plot_loss_history(loss_history, save_dir):
    plt.plot(loss_history)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(Path(save_dir, "loss.pdf"))
    plt.show()


def plot_samples(samples, sampler, save_dir):
    sample_grid = make_grid(samples, nrow=int(np.sqrt(len(samples))))

    plt.figure(figsize=(6, 6))
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(Path(save_dir, f"generation_{sampler.__name__}.pdf"))
    plt.show()
