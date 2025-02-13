# rd1-3 command
../../containers/shifty.sif ../../software/superfold/run_superfold.py --out_dir outputs --max_recycles 3 inputs/super_design.pdb --models 4

# rd4 command (N+1 oxyanion holes, phenylacetate substrate)
../../containers/af2_binder_design.sif ../../scripts/initial_guess.py -outdir outputs/ -contacts 20 -recycle 3 -pdbs inputs/refined_out_1_bn1_1.pdb
