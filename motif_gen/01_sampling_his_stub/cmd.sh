# for production runs, increase --max_random_rotamers and/or sampling in the cst file.
cd outputs; python ../../../scripts/invrotzyme/invrotzyme.py --cstfile ../inputs/1LNS_mu1.cst --params ../inputs/mu1.params --N_len 2 --C_len 2 --secstruct_per_cst H H --dunbrack_prob 0.6 --motif_for_cst 1:2:../inputs/1LNS.pdb --keep_his_tautomer 2:HIS_D --max_random_rotamers 5; cd ..
