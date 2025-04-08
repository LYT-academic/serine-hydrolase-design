# search for oxyanion hole residues interacting with O1 on the ligand
python ../../scripts/hb_search.py --pdb inputs/out_1_refined_0.pdb --outdir outputs --resis_to_fix 20,21,54 --ligand mu1 --target_atom O1 --donors --require_seqdist_from_target 8 --seqdist_target_resi 20 --resn_to_ignore RKH --params inputs/mu1.params
# search for Asp/Glu donors to histidine to extend the catalytic dyad to a triad
python ../../scripts/hb_search.py --pdb inputs/out_1_refined_0.pdb --outdir outputs --resis_to_fix 20,21,54 --ligand mu1 --target_resi 54 --target_atom ND1 --acceptors --require_seqdist_from_target 8 --seqdist_target_resi 20 --params inputs/mu1.params
