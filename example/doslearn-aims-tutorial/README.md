# Tutorial: predicting the global electronic density of states (DOS) with `doslearn` and `FHI-aims`

## Overview

This tutorial follows two parts: 1) data generation with `FHI-aims` and 2) model training with `doslearn`. Follow the instructions in the README files in subdirectories [`part-1-dft`](part-1-dft/README.md) and [`part-2-ml`](part-2-ml/README.md). The data used is a demonstration dataset of periodic silicon diamond systems. 
 
First, data is generated with `FHI-aims`. SCF calculations are converged to compute the self consistent solutions to the Kohn-Sham equations for each frame, along with, output of the eigenvalues on a dense k-grid.

Second, the DOS of each structure is computed via gaussian smearing of the eigenvalues obtained from the SCF calculations. The DOS for each frame is represented by Cubic Hermite Splines to facilitate the changing of the energy reference, which is an ill-defined quantity in bulk calculations. These splines form the dataset for training a machine learning model. In `doslearn`, an invariant power spectrum descriptor-based neural network is used to learn the mapping from nuclear coordinates to the DOS projected on a discretized energy grid. A model is trained iteratively over a number of epochs, optimizing the NN weights by backpropagation and gradient descent. Also optimized is an alignment vector that represents the energy reference for the DOS for each frame to aid model training.

## Supporting notebooks

Some basic and optional extras for each section of each tutorial README is provided in jupyter notebooks of the same name. These are intended to aid visualization and inspection of outputs.

## Setup

Follow the `rholearn` and `FHI-aims` installation instructions in the README of the main repository, [here](../../README.md).
