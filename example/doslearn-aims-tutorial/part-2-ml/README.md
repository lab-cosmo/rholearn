# Part 2: Train a surrogate model with `rholearn`


## 2.0: TLDR of requried commands

After copying [dft-options.yaml](../part-1-dft/dft-options.yaml) and [hpc-options.yaml](../part-1-dft/hpc-options.yaml) to this directory and modifying [ml-options.yaml](ml-options.yaml) as desired, the commands needed to train and evalute a model are below. For a full explanation of each, read on to the following sections.

```bash
# Copy dft-options.yaml and hpc-options.yaml into this directory, and modify ml-options.yaml if desired
# ...

# Train
doslearn_train

# Eval
doslearn_eval
```

Where each command can be wrapped in an HPC submission script.

## 2.1: Key parts of the training workflow

TODO

## 2.2: Specify ML settings

First copy files [dft-options.yaml](../part-1-dft/dft-options.yaml) and [hpc-options.yaml](../part-1-dft/hpc-options.yaml) into the current directory [part-2-ml](.). Next, inspect the user settings file [ml-options.yaml](ml-options.yaml) and edit the appropriate fields.

You can also inspect the default DFT settings, which can be printed with:
```python
import pprint
from rholearn.options import get_options

pprint.pprint(get_options("ml"))
```
Any of these can be modified by specification in the local file [ml-options.yaml](ml-options.yaml).

## 2.3: Train a model

**Training a model locally** can be done as follows:

```python
import rholearn

rholearn.doslearn.train()

# Alternatively: from the command line
doslearn_train
```

The file [hpc-options.yaml](hpc-options.yaml) is not used by the `rholearn.doslearn.train` function. Instead, **to run training on a cluster**, the one-line python command can be incorporated into an HPC run script. In this case, ensure that the calculation is run from within the `rho` conda environment. For slurm schedulers, this is done by running the script from the `rho` env with the `--get-user-env` flag in the submission script:

```bash
#!/bin/bash
#SBATCH --job-name=ml
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=40
#SBATCH --get-user-env

doslearn_train
```

## 2.4: Evaluate the model

The model can then be evaluated on the test set as follows:

```python
import rholearn

rholearn.doslearn.eval()

# Alternatively: from the command line
doslearn_eval
```
As evaluation of the model requires rebuilding the field in FHI-aims, the `rholearn.doslearn.eval` function requires specification of the local file [hpc-options.yaml](hpc-options.yaml).

Once running, information is logged to `part-2-ml/outputs/eval.log`. Model inference is performed: the nuclear coordinates of the test structures are transformed into an atom-centered density correlation descriptor, then passed through the neural network to yield the predicted DOS.
