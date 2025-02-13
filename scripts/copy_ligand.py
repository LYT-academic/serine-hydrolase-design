from SimplePdbLib import *
import sys

if sys.argv[1] == '--help' or sys.argv[1] == '-h':
    print('Usage: python copy_ligand.py [design model with ligand] [af2 model]')

input_pdb = sys.argv[1]
af2_pdb = sys.argv[2]

des_mdl = read_in_stubs_file(input_pdb)[0]
af2_mdl = read_in_stubs_file(af2_pdb)[0]
af2_name = af2_pdb.split('/')[-1]

ligand = des_mdl[-1]
af2_mdl.append(ligand)
write_models_to_file([af2_mdl], f'outputs/{af2_name}')
