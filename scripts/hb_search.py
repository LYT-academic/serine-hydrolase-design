import sys, os, glob, json, time
import pandas as pd
import numpy as np
import argparse 
from icecream import ic
parser = argparse.ArgumentParser()

# input arguments
parser.add_argument("--pdb", type=str, help="Path to pdb that needs a ligand")
parser.add_argument("--outdir", type=str, default='./', help="Path to output.")
parser.add_argument("--params", type=str, help="Params file. If not specified, will search in /projects/hydrolases folder.")
parser.add_argument("--resis_to_fix", type=str, default=None, help="Residues to fix. Usually catalytic residues.")
parser.add_argument("--ligand", type=str, help="Name of ligand")

# residue and atom in residue to hbond to
parser.add_argument("--target_resi", type=int, help="Residue index of residue to hbond with, default final residue.")
parser.add_argument("--require_seqdist_from_target", type=int, default=None, help="How far the new sidechain needs to be far from target in sequence (if applicable)")
parser.add_argument("--seqdist_target_resi", type=int, default=None, help="Different residue to be far away from if not from target (if applicable).")
parser.add_argument("--target_atom", type=str, help="Atom in residue to hbond with")
parser.add_argument("--donors",      action='store_true', default=False, help="Search for donors to target atom.")
parser.add_argument("--acceptors",   action='store_true', default=False, help="Search for acceptors to target atom.")
parser.add_argument("--resn_to_ignore", type=str, default='', help="String of 1-letter code residues you don't want as donors/acceptors")

parser.add_argument("--debug_rosetta", action='store_true', help="Don't mute rosetta output")
parser.add_argument("--debug", action='store_true', help="Save some helpful pdbs, NOTE turn off for production runs!!!")

args = parser.parse_args()
parser.set_defaults()
print("Using the following arguments:")
print(args)

assert any([args.donors, args.acceptors]), 'Must specify if you want acceptors, donors, or both.'

import pyrosetta
import pyrosetta.distributed.io
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts

import utils
import hbond_utils
import geometry
import the_one_util

#------------------------------------------------------------------------------------------
# load the ligand and initialize pyrosetta

if args.params:
    params = args.params
else:
    print('Attempting to find ligand params file automatically.')
    params = f'/projects/hydrolases/serine_hydrolase/params/{args.ligand}.params'
ic(params)

if args.debug_rosetta:
    pyrosetta.init(f'-run:preserve_header -beta -extra_res_fa {params} -holes::dalphaball /net/software/lab/scripts/enzyme_design/DAlphaBall.gcc')
else:
    pyrosetta.init(f'-mute all -run:preserve_header -beta -extra_res_fa {params} -holes::dalphaball /net/software/lab/scripts/enzyme_design/DAlphaBall.gcc')

#-------------------------------------------------------------------------------------------
# residue naming utils

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
chains = 'ABCDEFGH'

aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
aa_3_1['HIS_D'] = 'H'
#-------------------------------------------------------------------------------------------
# relax function

def relax_positions_with_constraints(pose, enzyme_cst_f='test.cst', positions=[], score_only=False):

    if positions == []:
        print('Nothing to relax.')
        return
    positions_str = ','.join([str(x) for x in positions])
    fr_call = '<Add mover="fastRelax"/>'
    enz_cst = '<Add mover="add_enz_csts"/>'
    sc_call = '<Add mover="score"/>'
    
    xml = f"""
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
          <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
          </ScoreFunction>
      </SCOREFXNS>
      <RESIDUE_SELECTORS>
          <Index name="positions" resnums="{positions_str}"/>
          <Not name="fixed" selector="positions"/>
      </RESIDUE_SELECTORS>
      <TASKOPERATIONS>
          <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
          <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" ex2="0"/>
          <IncludeCurrent name="ic"/>
          <OperateOnResidueSubset name="to_relax" selector="positions"> 
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="to_fix" selector="fixed"> 
              <PreventRepackingRLT/>
          </OperateOnResidueSubset>
      </TASKOPERATIONS>
      <MOVERS>    
          <AddOrRemoveMatchCsts name="add_enz_csts" cstfile="{enzyme_cst_f}" cst_instruction="add_new"/>
          <FastDesign name="fastRelax" scorefxn="sfxn_design" repeats="1" task_operations="ex1_ex2aro,ic,limitchi2,to_relax,to_fix" batch="false" ramp_down_constraints="false" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"> 
              <MoveMap name="MM" bb="0" chi="0" jump="0">
              </MoveMap>
          </FastDesign>
          <ScoreMover name="score" scorefxn="sfxn_design" />
      </MOVERS>
      <FILTERS>
          <EnzScore name="cst_score" scorefxn="sfxn_design" confidence="0" whole_pose="1" score_type="cstE" energy_cutoff="100.0"/>
      </FILTERS>
      <SIMPLE_METRICS>
      </SIMPLE_METRICS>
      <PROTOCOLS>
         {enz_cst if not score_only else ''}
         {fr_call if not score_only else sc_call}
         <Add filter="cst_score"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    df_scores = pd.DataFrame.from_records([packed_pose.scores])

    return pose, df_scores
#---------------------------------------------------------------------------------------------------

# # look for outputs and quit if detected
# outs = glob.glob(f"{args.outdir}/{args.pdb.split('/')[-1].replace('.pdb',f'_hbsearch_*.pdb')}")
# if len(outs) > 0:
#     sys.exit('found existing outputs')


# TODO: script assumes there is a ligand in the pdb

# 0 - make a set of constraint files for each atom type
#CONSTRAINT_FILE_PATH = '/projects/hydrolases/serine_hydrolase/shared_data/hbsearch/hbsearch_ATOMTYPE_TARGET.cst'
CONSTRAINT_FILE_PATH = 'inputs/hbsearch_ATOMTYPE_TARGET.cst'

# 1 - build set of amino acids to mutate to based on donors/acceptors flag

donors = ['S','T','Y','R','H','K','N','Q','W']
acceptors = ['D','E']

atom_types = {'D': 'OOC',
              'E': 'OOC',
              'S': 'OH',
              'T': 'OH',
              'Y': 'OH',
             #'R': 'um',
              'H': 'Ntrp',
              'K': 'Nlys',
              'N': 'NH2O',
              'Q': 'NH2O',
              'W': 'Ntrp'}

mutate_to = []
if args.donors:
    mutate_to += donors
if args.acceptors:
    mutate_to += acceptors
mutate_to = [res for res in mutate_to if res not in args.resn_to_ignore]

# 2 - find the residues with CAs near the atom of interest

# load the pose
pose = utils.load_diffusion_pose(args.pdb)

# get the target atom neighborhood
target_resi = args.target_resi
if not args.target_resi:
    target_resi = pose.size()

target_resi_chain = chains[pose.residue(target_resi).chain()]
target_resi_pdb = pose.pdb_info().number(target_resi)
target_resi_name = pose.residue(target_resi).name3()

target_resi_chain = pose.pdb_info().chain(target_resi)

ic(target_resi_name)
ic(target_resi_chain)
ic(target_resi, target_resi_pdb)

neighbor_resi = the_one_util.atom_neighborhood(pose, target_resi, args.target_atom)


# remove any fixed residues from the neighborhood
fixed_resis = [int(i) for i in args.resis_to_fix.split(',')]
neighbor_resi = [i for i in neighbor_resi if i not in fixed_resis]
ic(neighbor_resi)
ic(fixed_resis)

# 3 - mutate all nearby residues to alanine (except fixed ones)

for resi in neighbor_resi:
    pose = the_one_util.mutate(pose, resi, 'ALA')

# just score the pose to use as a baseline for fa_rep
scored_pose, norelax_scores = relax_positions_with_constraints(pose, positions=[1], score_only=True)
initial_fa_rep = round(norelax_scores['fa_rep'].squeeze(), 1)
ic(initial_fa_rep)

# 4 - for loop on every position: mutate->relax->evaluate

for position in neighbor_resi:

    if args.require_seqdist_from_target:
        seqdist_target = args.seqdist_target_resi if args.seqdist_target_resi else target_resi
        ic(seqdist_target)
        if abs(seqdist_target - position) <= args.require_seqdist_from_target:
            print(f'Skipping position {position} because it is too close to target in sequence')
            continue

    print()
    print(f'Mutating position {position}.')

    for resn in mutate_to:

        print()
        print(f'\tNext mutation: {resn}')

        mutated_pose = pose.clone()

        # mutate in the residue of interest
        mutated_pose = the_one_util.mutate(mutated_pose, position, aa_1_3[resn])

        # create header and add the header to the protein
        cst_header = f"REMARK 666 MATCH TEMPLATE {target_resi_chain} {mutated_pose.residue(target_resi).name3()}    {target_resi_pdb} MATCH MOTIF {chains[mutated_pose.residue(position).chain()-1]} {aa_1_3[resn]}  {position}  1  1               \n"
        as_str = pyrosetta.distributed.io.to_pdbstring(mutated_pose)

        # remove any existing remark 666 lines to avoid incompatibilities with cst file
        as_str = '\n'.join([l for l in as_str.split('\n') if 'REMARK 666' not in l])
        as_str = cst_header + as_str
        mutated_pose = pyrosetta.Pose()
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(mutated_pose, as_str)

        if args.debug:
            mutated_pose.dump_pdb(f'mutated_w_header_{position}{resn}.pdb')

        # locate the constraint file
        atom_type = atom_types[resn]
        cstf = CONSTRAINT_FILE_PATH.replace('ATOMTYPE', atom_type).replace('TARGET', target_resi_name)

        # do a speedy relax of just the residue of interest
        mutated_pose, scores = relax_positions_with_constraints(mutated_pose, enzyme_cst_f=cstf, positions=[position])
        cst_score = round(scores['cst_score'].squeeze(), 1)
        fa_rep = round(scores['fa_rep'].squeeze(), 1)
        print('\tcst_score      ', cst_score)
        print('\tdelta fa_rep   ', round(fa_rep - initial_fa_rep, 1))

        if args.debug:
            mutated_pose.dump_pdb(f'relaxed_{position}{resn}.pdb')

        # check for an hbond, require low fa_rep to avoid accepting clashing stuff.
        if cst_score == 0 and (fa_rep - initial_fa_rep < 100):
            print('\tFound hbond.')

            # save the pdb and the hbond residue location / residue name
            os.makedirs(args.outdir, exist_ok=True)
            pdbname = f"{args.outdir}/{args.pdb.split('/')[-1].replace('.pdb',f'_hbsearch_{position}{resn}.pdb')}"
            mutated_pose.dump_pdb(pdbname)

