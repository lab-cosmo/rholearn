# Tutorial: predicting electron densities with `rholearn` and `FHI-aims`

## Overview

This tutorial follows two parts: 1) data generation with `FHI-aims` and 2) model training with `rholearn`. Follow the instructions in the README files in subdirectories [`part-1-dft`](part-1-dft/README.md) and [`part-2-ml`](part-2-ml/README.md). 

First, data is generated with `FHI-aims` for a subset of structures from the QM7 database that contain atom types H, C, O, N. This involves a two step process: a) converging SCF calculations to compute the self consistent electron density for each frame, then b) decomposing the electron density scalar field onto a fitted basis set.

Second, the reference data output from the first step, in the form of fitting coefficients, projections, and overlap matrices, form the dataset for training a machine learning model. In `rholearn`, arbitrary descriptor-based equivariant neural networks can be used to learn the mapping from nuclear coordinates to basis set expansion coefficients. 

Typically, the descriptor is an equivariant power spectrum (or $\lambda$ -SOAP), which is passed through a linear layer or small multi-layer perceptron to transform it into a vector of predicted coefficients. A model is trained iteratively over a number of epochs, optimizing the NN weights by backpropagation and gradient descent.

## Supporting notebooks

Some basic and optional extras for each section of each tutorial README is provided in jupyter notebooks of the same name. These are intended to aid visualization and inspection of outputs.

## Setup

Follow the `rholearn` and `FHI-aims` installation instructions in the README of the main repository, [here](../../README.md).
