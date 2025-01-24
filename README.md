# *rholearn*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13891847.svg)](https://doi.org/10.5281/zenodo.13891847)

`metatensor-torch` workflows for training descriptor-based equivariant neural networks to predict at DFT-level accuracy:

1) real-space electronic density scalar fields decomposed on a basis (molecular & periodic systems)
2) electronic density of states (DOS) (periodic systems)


**Authors**: 
* [Joseph W. Abbott](https://linktr.ee/josephabbott), PhD Student @ Lab COSMO, EPFL
* Wei Bin How, PhD Student @ Lab COSMO, EPFL

**Note:** under active development, breaking changes are likely!

![rholearn workflow summary](example/assets/rholearn.png)

# Background

## Real-space electronic densities

Electronic densities, such as the electron density and local density of states, are central quantities in understanding the electronic properties of molecules and materials on the atomic scale. First principles quantum simulations such as density-functional theory (DFT) are able to accurately predict such fields as a linear combination of single-particle solutions to the Kohn-Sham equations. While reliable and accurate, such methods scale unfavourably with the number of electrons in the system. 

Machine learning methods offer a complementary solution to probing the electronic structure of matter on the atomic scale. With a sufficiently expressive model, one can learn the mapping between nuclear geometry and real-space electronic density and predict such quantities with more favourable scaling. Typically, predictions can be used to accelerate DFT by providing initial guesses, or directly probe electronic structure.

There are many approaches to learn the aforementioned mapping. In the ***density fitting approach***, the real-space target electronic density $\rho^{\text{DFT}}(\mathbf{r})$ is decomposed onto a linear atom-centered basis set:

$$
\rho^{\text{DFT}}(\mathbf{r}) \approx \rho^{\text{RI}}(\mathbf{r}) = \sum_{b} d_b^{\text{RI}} \ \varphi_b^{\text{RI}}(\mathbf{r})
$$

where $\rho^{\text{RI}}(\mathbf{r})$ the basis set approximation to it, $\varphi_b^{\text{RI}}(\mathbf{r})$ are fitted basis functions (each a product of radial function and spherical harmonics) generated by the *resolution-of-the-identity* (RI) approach, and $d_b^{\text{RI}}$ are the coefficients that minimize the basis set expansion error for the given basis set definition.

An equivariant model is then trained to predict coefficients $d_b^{\text{ML}}$ that reconstruct a density in real-space, ideally minimising the generalisation error on the real-space DFT densities of a test set.

For one of the original workflows for predicting the electron density under the density-fitting framework, readers are referred to [*SALTED*](https://github.com/andreagrisafi/SALTED). This uses a symmetry-adapted Gaussian process regression (SA-GPR) method via sparse kernel ridge regression to learn and predict $d_b^{\text{ML}}$.

## Electronic density of states (DOS)

The electronic density of states (DOS) provides information regarding the distribution of available electronic states in a material. With the DOS of a material, one is able to infer many optical and electronic properties of a material, such as its electrical conductivity, bandgap and absorption spectra. This allows the DOS to be relevant for material design as a tool to screen potential material candidates. The DOS is typically computed using DFT, but as mentioned above, DFT is prohibitively expensive for large and complex systems.

Machine learning has also been applied to the DOS and a variety of representatinos have been developed for the DOS. Thus far, there have been three main approaches, 1) Projecting the DOS on a discretized energy grid, 2) Projecting the integrated DOS on a discretized energy grid and, 3) Decomposing the DOS using Principal Component Analysis (PCA). Unlike the electronic density, the DOS is invariant to rotations and thus an invariant model can be employed to predict the DOS under any of the three representations.

# Goals

`rholearn` also operates under the density fitting approach. The nuclear coordinates $\to$ electonic density mapping is learned *via* a feature-based equivariant neural network whose outputs are the predicted coefficients. Currently, `rholearn` is integrated with the electronic structure code `FHI-aims` for both data generation and building of real-space fields from predicted coefficients. `rholearn` aims to improve the scalability of the density-fitting approach to learning electronic densities. 

`doslearn` represents the DOS by projecting it on a discretized energy grid. Additionally, a locality ansatz is employed whereby the global DOS of a structure, is expressed as a sum of local contributions from each atomic environment.

Both are built on top of a modular software ecosystem, with the following packages forming the main components of the workflow:

* **`metatensor`** ([GitHub](https://github.com/lab-cosmo/metatensor)) is used as the self-describing block-sparse data storage format, wrapping multidimensional tensors with metadata. Subpackages `metatensor-operations` and `metatensor-learn` are used to provide convenient sparse operations and ML building blocks respectively that operate on the `metatensor.TensorMap` object.
* **`featomic`** ([GitHub](https://github.com/metatensor/featomic)) is used to transform the nuclear coordinates into local equivariant descriptors that encode physical symmetries and geometric information for input into the neural network.
* **`PyTorch`** is used as the learning framework, allowing definition of arbitrarily complex neural networks that can be trained by minibatch gradient descent.

Leveraging the speed- and memory-efficient operations of `torch`, and using building on top of `metatensor` and `featomic`, descriptors, models, and learning methodologies can be flexibly prototyped and customized for a specific learning task.


# Getting Started

### Installing `rholearn`

With a working `conda` installation, first set up an environment:
```bash
conda create -n rho python==3.12
conda activate rho
```
Then clone and install `rholearn`:
```bash
git clone https://github.com/lab-cosmo/rholearn.git
cd rholearn
# Specify CPU-only torch
pip install --extra-index-url https://download.pytorch.org/whl/cpu .
```

Running `tox` from the top directory will run linting and formatting.
To run some tests (currently limited to testing `rholearn.loss`), run `pytest tests/rholearn/loss.py`.

### Installing `FHI-aims`

For generating reference data, using the `aims_interface` of `rholearn`, a working installation of **`FHIaims >= 240926`** is required. FHI-aims is not open source but is free for academic use. Follow the instructions on their website [fhi-aims.org/get-the-code](https://fhi-aims.org/get-the-code/) to get and build the code. The end result should be an executable, compiled for your specific system.

There are also useful tutorials on the basics of running `FHI-aims` [here](https://fhi-aims-club.gitlab.io/tutorials/basics-of-running-fhi-aims/).


### Basic usage

In a run directory, user-options are defined in YAML files named ["dft-options.yaml"](example/options/dft-options.yaml), ["hpc-options.yaml"](example/options/hpc-options.yaml), and ["ml-options.yaml"](example/options/ml-options.yaml). Any options specified in these files overwrite the defaults.

Default options can be found in the [rholearn/options/](rholearn/options) directory, and some templates for user options can be found in the [examples/options/](example/options) directory.

#### `rholearn`

Data can be generated with the following:

```bash
rholearn_run_scf  # run SCF with FHI-aims

rholearn_process_scf  # process SCF outputs

rholearn_setup_ri_fit  # setup RI fitting calculation

rholearn_run_ri_fit  # run RI fitting with FHI-aims

rholearn_process_ri_fit  # process RI outputs
```

and model training and evaluation run with:

```bash
rholearn_train  # train model

rholearn_eval  # evaluate model
```

#### `doslearn`

Data can be generated with the following:

```bash
doslearn_run_scf  # run SCF with FHI-aims

doslearn_process_scf  # process SCF outputs
```

and model training and evaluation run with:

```bash
doslearn_train  # train model

doslearn_eval  # evaluate model
```


### Tutorial 

For a more in-depth walkthrough of the functionality, see the following tutorials:

1. [rholearn tutorial](example/rholearn-aims-tutorial/README.md) on data generation using `FHI-aims` and model training using `rholearn` to predict the electron density decomposed on a basis.
2. [doslearn tutorial](example/doslearn-aims-tutorial/README.md) on data generation using `FHI-aims` and model training using `doslearn` to predict the electron density of states.


# Citing this work

```bib
@software{abbott_2024_13891847,
  author       = {Abbott, Joseph W. and
                  How, Wei Bin and
                  Fraux, Guillaume and
                  Ceriotti, Michele},
  title        = {lab-cosmo/rholearn: rholearn v0.1.0},
  month        = oct,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.13891847},
  url          = {https://doi.org/10.5281/zenodo.13891847}
}

@article{PhysRevMaterials.9.013802,
  author       = {How, Wei Bin and
                  Chong, Sanggyu and
                  Grasselli, Federico and
                  Huguenin-Dumittan, Kevin K. and
                  Ceriotti, Michele},
  title        = {Adaptive energy reference for machine-learning models of the electronic density of states},
  journal      = {Phys. Rev. Mater.},
  volume       = {9},
  issue        = {1},
  pages        = {013802},
  numpages     = {10},
  year         = {2025},
  month        = {Jan},
  publisher    = {American Physical Society},
  doi          = {10.1103/PhysRevMaterials.9.013802},
  url          = {https://link.aps.org/doi/10.1103/PhysRevMaterials.9.013802},
}

```
