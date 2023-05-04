# ENSAE Deep Learning

[![Code quality](https://github.com/baptiste-pasquier/ensae-deep-learning/actions/workflows/quality.yml/badge.svg)](https://github.com/baptiste-pasquier/ensae-deep-learning/actions/workflows/quality.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baptiste-pasquier/ensae-deep-learning/)

The project reproduces the results from the following paper: "*Score-Based Generative Modeling through Stochastic Differential Equations*" [[1]](#1) and uses the associated code (see [Acknowledgments](#acknowledgments)).

- MNIST notebook: [link](notebooks/notebook_mnist.ipynb)
- EMNIST notebook: [link](notebooks/notebook_emnist.ipynb)
- Fashion-MNIST notebook: [link](notebooks/notebook_fashion.ipynb)
- Flowers 102 notebook: [link](notebooks/notebook_flowers.ipynb)

An "Open in Colab" button is present on each notebook to open and run it directly on Google Colab. For a fast execution you can reduce the number of epochs in the config (`"n_epochs"`).

Opening the notebooks on Github allow you to view a complete training and the corresponding generated images with a large number of epochs.

## Installation

1. Clone the repository
```bash
git clone https://github.com/baptiste-pasquier/ensae-deep-learning
```

2. Install the project
- With `poetry` ([installation](https://python-poetry.org/docs/#installation)):
```bash
poetry install
```
- With `pip` :
```bash
pip install -e .
```

3. (Optional) Install Pytorch CUDA
```bash
poe torch_cuda
```

## References

<a id="1">[1]</a> Song, Yang, et al. "[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)". arXiv:2011.13456 (2020).

## Acknowledgments

The project relies mainly on the following repositories:
- https://github.com/yang-song/score_sde_pytorch
- https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing
