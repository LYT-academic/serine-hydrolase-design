export PYTHONPATH=/work/lyt/serine-hydrolase-design/software/ca_rf_diffusion
python ../../software/ca_rf_diffusion/rf_diffusion/run_inference.py \
--config-name RFdiffusion_CA_inference \
inference.num_designs=1 \
inference.ckpt_path="../../software/ca_rf_diffusion/checkpoints/ca_rfd_diffusion.pt" \
inference.output_prefix="outputs_test/out" \
inference.input_pdb="inputs/simple_theozyme.pdb" \
inference.ligand="mu1" \
inference.ij_visible="abc" \
contigmap.contigs="['18,A1-3,31,B5-7,45']" \
motif_only_2d=true \
diffuser.r3.noise_scale=0.05
