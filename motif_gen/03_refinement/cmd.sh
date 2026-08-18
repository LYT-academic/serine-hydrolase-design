#!/bin/bash 
export PYTHONPATH=~/serine-hydrolase-design/software/ca_rf_diffusion

# Note: you need to define these 
pdb="../03_refinement/inputs_test/out_0.pdb"
CKPT="../../software/ca_rf_diffusion/checkpoints/ca_rfd_refinement.pt"

# This is an example of supplying extra flags to override the .yaml file. 
# Note that the inference.ligand needs to match your specific ligand in the pdb files.
# 'mu2' is here as an example because that's what the diffusion step example also uses.
python ../../software/ca_rf_diffusion/rf_diffusion/run_inference.py \
    --config-name=test_ca_rfd_refinement \
    inference.num_designs=4 \
    inference.input_pdb=$pdb \
    inference.ligand='mu1' \
    inference.ckpt_path=$CKPT
