#!/usr/bin/env bash

set -eux

# ===== 'Cube-Toolz' for reading and manipulating cube files
pip uninstall -y cube_tools
pip install git+https://github.com/funkymunkycool/Cube-Toolz.git

# ===== Uninstalls, clear caches
pip uninstall -y torch rascaline rascaline-torch metatensor metatensor-core metatensor-operations metatensor-torch metatensor-learn

# ===== PyTorch, rascaline, rascaline-torch with CPU-only torch
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.4.1
pip install metatensor[torch]

pip install --no-build-isolation --no-deps "rascaline @ git+https://github.com/luthaf/rascaline@fb7f363cd23498e97e24e67f37c6c49cc7826884"
pip install --no-build-isolation --no-deps "rascaline-torch @ git+https://github.com/luthaf/rascaline@fb7f363cd23498e97e24e67f37c6c49cc7826884#subdirectory=python/rascaline-torch"
