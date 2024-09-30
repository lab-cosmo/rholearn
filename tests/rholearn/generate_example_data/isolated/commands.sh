# ===== Run these commands separately and in order to generate the example data

# Run SCF
# python -c 'from rholearn.aims_interface import scf; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; scf.run_scf(DFT_SETTINGS, HPC_SETTINGS);'

# Process SCF
# python -c 'from rholearn.aims_interface import scf; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; scf.process_scf(DFT_SETTINGS, HPC_SETTINGS);'

# Setup RI
# python -c 'from rholearn.aims_interface import ri_fit; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; ri_fit.set_up_ri_fit_sbatch(DFT_SETTINGS, HPC_SETTINGS);'

# Run RI
# python -c 'from rholearn.aims_interface import ri_fit; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; ri_fit.run_ri_fit(DFT_SETTINGS, HPC_SETTINGS);'

# Process RI
# python -c 'from rholearn.aims_interface import ri_fit; from dft_settings import DFT_SETTINGS; from hpc_settings import HPC_SETTINGS; ri_fit.process_ri_fit(DFT_SETTINGS, HPC_SETTINGS)'
