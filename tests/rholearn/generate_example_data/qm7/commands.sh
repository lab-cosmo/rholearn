# ===== Run these commands separately and in order to generate the example data

# Run SCF
python -c 'from rholearn.aims_interface import scf; scf.run_scf()'

# Process SCF
python -c 'from rholearn.aims_interface import scf; scf.process_scf()'

# Setup RI
python -c 'from rholearn.aims_interface import ri_fit; ri_fit.set_up_ri_fit_sbatch()'

# Run RI
python -c 'from rholearn.aims_interface import ri_fit; ri_fit.run_ri_fit()'

# Process RI
python -c 'from rholearn.aims_interface import ri_fit; ri_fit.process_ri_fit()'

# Setup RI rebuild
python ri_rebuild_setup.py

# Run RI rebuild
python -c 'from rholearn.aims_interface import ri_rebuild; ri_rebuild.run_ri_rebuild()'

# Process RI rebuild
python ri_rebuild_process.py