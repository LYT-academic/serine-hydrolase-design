import pyrosetta
import pandas as pd
import sys
import numpy as np
import os
import random
import json
import string
import math

sys.path.append("../../software")
from pyrosetta.rosetta.core.select.residue_selector import (
    AndResidueSelector,
    ChainSelector,
    NotResidueSelector,
    ResidueIndexSelector,
    OrResidueSelector
)
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
from SimplePdbLib import *

def align_n_add_ligand(pose, ligand_pose, pose_resi, ligand_pose_resi):
    print('Adding ligand')
    # pose, ligand pose = pose without ligand, pose to copy ligand from
    # pose_resi, ligand_pose_resi = residue idxs to align from each pose

    # align the stub and ligand motif by the nucleophilic serine backbone atoms - works great!
    align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
    aln_atoms = ['N', 'CA', 'C', 'O']
    for pose_i, stub_i in zip(pose_resi, ligand_pose_resi):
        res_pose_i = pose.residue(int(pose_i))
        res_stub_i = ligand_pose.residue(int(stub_i))
        for n in aln_atoms:
            pose_atom_idx = res_pose_i.atom_index(n)
            atom_id_pose = pyrosetta.rosetta.core.id.AtomID(pose_atom_idx, pose_i)
            stub_atom_idx = res_stub_i.atom_index(n)
            atom_id_stub = pyrosetta.rosetta.core.id.AtomID(stub_atom_idx, stub_i)
            align_map[atom_id_stub] = atom_id_pose
    rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(ligand_pose, pose, align_map)

    # take out the old ligand from the pose, copy the new ligand into the aligned design pose
    for chain in list(ligand_pose.split_by_chain()):
        if chain.size() == 1 and chain.residue(1).is_ligand():
            ligand = chain
    if not ligand:
        sys.exit('pose to copy ligand from has no ligand')

    # debug
    #pose.dump_pdb(f'aligned.pdb')
    #ligand_pose.dump_pdb(f'stub_aligned.pdb')

    pose = list(pose.split_by_chain())[0]
    pyrosetta.rosetta.core.pose.append_pose_to_pose(pose, ligand, new_chain=True)
    return pose

def add_matcher_lines_to_pose(pose, catres, ligand, blocks=[1,2,3,4]):
    """
    STOP! did you add -run:preserve_header into your pyrosetta init? if not, this function will not work.
    """
    def make_line(i, u_resi, d_resi, upstream_ligand=False):
        if upstream_ligand:
            line = f"REMARK 666 MATCH TEMPLATE X {ligand}    0 MATCH MOTIF A {pose.residue(d_resi).name3()}  {d_resi}  {blocks[i]}  1               "
        else:
            line = f"REMARK 666 MATCH TEMPLATE A {pose.residue(u_resi).name3()}    {u_resi} MATCH MOTIF A {pose.residue(d_resi).name3()}  {d_resi}  {blocks[i]}  1               "
        i += 1
        return line, i

    ligand_resi = pose.size()
    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")
    new_pdb = []
    i = 0
    for l in pdbff:
        if "HEADER" in l:
            new_pdb.append(l)
            # ligand-serine
            line, i = make_line(i, ligand_resi, catres['ser'], upstream_ligand=True)
            new_pdb.append(line)
            # serine-histidine
            line, i = make_line(i, catres['ser'], catres['his'], upstream_ligand=False)
            new_pdb.append(line)
            if 'acc' in catres.keys():
              # histidine-proton acceptor
              line, i = make_line(i, catres['his'], catres['acc'], upstream_ligand=False)
              new_pdb.append(line)
            # ligand-oxyanion hole
            line, i = make_line(i, ligand_resi, catres['oxy'], upstream_ligand=True)
            new_pdb.append(line)
        elif 'REMARK 666' in l:
            continue
        else:
            new_pdb.append(l)
    #print('\n'.join(new_pdb))
    pose2 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb))
    return pose2

def fastdesign(pose, design, repack, fix, lig_resi, enzyme_cst_f):
    # design, repack and fix are strings of comma separated residues to fix during design
    print('Starting fastdesign.')
    xml = f"""
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
          <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="arg_cation_pi" weight="3"/>
              <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
              <Reweight scoretype="dihedral_constraint" weight="1.0"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="coordinate_constraint" weight="1.0"/>
              <Reweight scoretype="pro_close" weight="0"/>
          </ScoreFunction>
      </SCOREFXNS>
      <RESIDUE_SELECTORS>
          <Chain name="chainA" chains="A"/>
          <Not name="chainB" selector="chainA" />
          <Index name="design" resnums="{design}"/> 
          <Index name="repack" resnums="{repack}"/> 
          <Index name="fix"    resnums="{fix}"/> 
          <Index name="ligand" resnums="{lig_resi}"/> 
      </RESIDUE_SELECTORS>
      <TASKOPERATIONS>
          <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
          <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" ex2="0"/>
          <IncludeCurrent name="ic"/>
          <OperateOnResidueSubset name="no_new_cys" selector="design">
              <RestrictAbsentCanonicalAASRLT aas="ADEFGHIKLMNPQRSTVWY"/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="to_repack" selector="repack"> 
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="to_fix" selector="fix"> 
              <PreventRepackingRLT/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="fix_lig" selector="ligand"> 
              <PreventRepackingRLT/>
          </OperateOnResidueSubset>
      </TASKOPERATIONS>
      <MOVERS>    
          <AddOrRemoveMatchCsts name="add_enz_csts" cstfile="{enzyme_cst_f}" cst_instruction="add_new"/>
          <FastDesign name="fastDesign" scorefxn="sfxn_design" repeats="1" task_operations="ex1_ex2aro,ic,limitchi2,no_new_cys,to_fix,to_repack" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"> 
              <MoveMap name="MM" bb="1" chi="1" jump="1"/>
          </FastDesign>
          <ClearConstraintsMover name="rm_csts" />
      </MOVERS>
      <FILTERS>
          <SimpleHbondsToAtomFilter name="O1_hbond" n_partners="1" hb_e_cutoff="-0.5" target_atom_name="O1" res_num="{lig_resi}" scorefxn="sfxn_design" confidence="0"/>
          <ContactMolecularSurface name="contact_molecular_surface" use_rosetta_radii="true" distance_weight="0.5" target_selector="chainA" binder_selector="chainB" confidence="0"/>
          <EnzScore name="cst_score" scorefxn="sfxn_design" confidence="0" whole_pose="1" score_type="cstE" energy_cutoff="100.0"/>
      </FILTERS>
      <SIMPLE_METRICS>
      </SIMPLE_METRICS>
      <PROTOCOLS>
         <Add mover="add_enz_csts"/>
         <Add mover="fastDesign"/>
         <Add filter="contact_molecular_surface"/>
         <Add filter="O1_hbond"/>
         <Add filter="cst_score"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    df_scores = pd.DataFrame.from_records([packed_pose.scores])

    return pose, df_scores

def fastdesign_norepack(pose, design, fix, lig_resi, enzyme_cst_f):
    print('Starting fastdesign.')
    # design and fix are strings of comma separated residues to fix during design

    xml = f"""
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
          <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="arg_cation_pi" weight="3"/>
              <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
              <Reweight scoretype="dihedral_constraint" weight="1.0"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="coordinate_constraint" weight="1.0"/>
              <Reweight scoretype="pro_close" weight="0"/>
          </ScoreFunction>
      </SCOREFXNS>
      <RESIDUE_SELECTORS>
          <Chain name="chainA" chains="A"/>
          <Not name="chainB" selector="chainA" />
          <Index name="design" resnums="{design}"/> 
          <Index name="fix"    resnums="{fix}"/> 
          <Index name="ligand" resnums="{lig_resi}"/> 
      </RESIDUE_SELECTORS>
      <TASKOPERATIONS>
          <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
          <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" ex2="0"/>
          <IncludeCurrent name="ic"/>
          <OperateOnResidueSubset name="no_new_cys" selector="design">
              <RestrictAbsentCanonicalAASRLT aas="ADEFGHIKLMNPQRSTVWY"/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="to_fix" selector="fix"> 
              <PreventRepackingRLT/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="fix_lig" selector="ligand"> 
              <PreventRepackingRLT/>
          </OperateOnResidueSubset>
      </TASKOPERATIONS>
      <MOVERS>    
          <AddOrRemoveMatchCsts name="add_enz_csts" cstfile="{enzyme_cst_f}" cst_instruction="add_new"/>
          <FastDesign name="fastDesign" scorefxn="sfxn_design" repeats="1" task_operations="ex1_ex2aro,ic,limitchi2,no_new_cys,to_fix" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"> 
              <MoveMap name="MM" bb="1" chi="1" jump="1"/>
          </FastDesign>
          <ClearConstraintsMover name="rm_csts" />
      </MOVERS>
      <FILTERS>
          <SimpleHbondsToAtomFilter name="O1_hbond" n_partners="1" hb_e_cutoff="-0.5" target_atom_name="O1" res_num="{lig_resi}" scorefxn="sfxn_design" confidence="0"/>
          <ContactMolecularSurface name="contact_molecular_surface" use_rosetta_radii="true" distance_weight="0.5" target_selector="chainA" binder_selector="chainB" confidence="0"/>
          <EnzScore name="cst_score" scorefxn="sfxn_design" confidence="0" whole_pose="1" score_type="cstE" energy_cutoff="100.0"/>
      </FILTERS>
      <SIMPLE_METRICS>
      </SIMPLE_METRICS>
      <PROTOCOLS>
         <Add mover="add_enz_csts"/>
         <Add mover="fastDesign"/>
         <Add filter="contact_molecular_surface"/>
         <Add filter="O1_hbond"/>
         <Add filter="cst_score"/>

    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    df_scores = pd.DataFrame.from_records([packed_pose.scores])

    return pose, df_scores

def remove_constraints(pose):
    xml = f"""
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
          <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="arg_cation_pi" weight="3"/>
              <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
              <Reweight scoretype="dihedral_constraint" weight="1.0"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="coordinate_constraint" weight="1.0"/>
              <Reweight scoretype="pro_close" weight="0"/>
          </ScoreFunction>
      </SCOREFXNS>
      <RESIDUE_SELECTORS>
      </RESIDUE_SELECTORS>
      <TASKOPERATIONS>
      </TASKOPERATIONS>
      <MOVERS>    
          <ClearConstraintsMover name="rm_csts" />
      </MOVERS>
      <FILTERS>
      </FILTERS>
      <SIMPLE_METRICS>
      </SIMPLE_METRICS>
      <PROTOCOLS>
         <Add mover="rm_csts"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    df_scores = pd.DataFrame.from_records([packed_pose.scores])
    return pose, df_scores

def fastrelax(pose, design, fix, lig_resi, enzyme_cst_f):
    print('Starting fastrelax.')
    # fix is string of comma separated residues to fix during design
    xml = f"""
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
          <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="arg_cation_pi" weight="3"/>
              <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
              <Reweight scoretype="dihedral_constraint" weight="1.0"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="coordinate_constraint" weight="1.0"/>
              <Reweight scoretype="pro_close" weight="0"/>
          </ScoreFunction>
      </SCOREFXNS>
      <RESIDUE_SELECTORS>
          <Chain name="chainA" chains="A"/>
          <Not name="chainB" selector="chainA" />
          <Index name="design" resnums="{design}"/> 
          <Index name="fix"    resnums="{fix}"/> 
          <Index name="ligand" resnums="{lig_resi}"/> 

          <Not name="notfixed" selector="fix"/>
          <And name="chA_notfixed" selectors="chainA,notfixed"/>
      </RESIDUE_SELECTORS>
      <TASKOPERATIONS>
          <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
          <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" ex2="0"/>
          <IncludeCurrent name="ic"/>
          <OperateOnResidueSubset name="no_new_cys" selector="design">
              <RestrictAbsentCanonicalAASRLT aas="ADEFGHIKLMNPQRSTVWY"/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="to_fix" selector="fix"> 
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="fix_lig" selector="ligand"> 
              <PreventRepackingRLT/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="to_repack" selector="chA_notfixed"> 
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>
      </TASKOPERATIONS>
      <MOVERS>    
          <AddOrRemoveMatchCsts name="add_enz_csts" cstfile="{enzyme_cst_f}" cst_instruction="add_new"/>
          <FastDesign name="fastRelax" scorefxn="sfxn_design" repeats="1" task_operations="ex1_ex2aro,ic,limitchi2,no_new_cys,to_fix,fix_lig,to_repack" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"> 
              <MoveMap name="MM" bb="1" chi="1" jump="1"/>
          </FastDesign>
          <ClearConstraintsMover name="rm_csts" />
      </MOVERS>
      <FILTERS>
          <SimpleHbondsToAtomFilter name="O1_hbond" n_partners="1" hb_e_cutoff="-0.5" target_atom_name="O1" res_num="{lig_resi}" scorefxn="sfxn_design" confidence="0"/>
          <ContactMolecularSurface name="contact_molecular_surface" use_rosetta_radii="true" distance_weight="0.5" target_selector="chainA" binder_selector="chainB" confidence="0"/>
          <EnzScore name="cst_score" scorefxn="sfxn_design" confidence="0" whole_pose="1" score_type="cstE" energy_cutoff="100.0"/>
      </FILTERS>
      <SIMPLE_METRICS>
      </SIMPLE_METRICS>
      <PROTOCOLS>
         <Add mover="add_enz_csts"/>
         <Add mover="fastRelax"/>
         <Add filter="contact_molecular_surface"/>
         <Add filter="O1_hbond"/>
         <Add filter="cst_score"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    df_scores = pd.DataFrame.from_records([packed_pose.scores])

    return pose, df_scores
    return

def remove_pose_matcher_lines(pose):
    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")
    new_pdb = []
    for l in pdbff:
        if "REMARK" in l:
            continue
        new_pdb.append(l)
    pose2 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb))
    return pose2


def load_diffusion_pose(pdb):
    """
    A function for reading diffusion outputs into rosetta. These outputs cause
    problems due overlapping sidechain atoms in unconstrained regions. These
    sidechain atoms are automatically removed.

    Inputs:
    path to diffusion pdb

    Returns:
    pyrosetta pose of pdb

    """

    pdbname = pdb.split('/')[-1]
    model = read_in_stubs_file(pdb)[0]
    new_model = []

    for res in model:
        keep_atoms = []
        prev_atom = None

        for atom in res.atom_records:
            # keep backbone atoms (these do not cause issues)
            if atom.name in ['C','N','CA','O','CB']:
                keep_atoms.append(atom)
                continue
            # if we find atoms with identical coordinates, remove them.
            if prev_atom:
                if (atom.x == prev_atom.x) and (atom.y == prev_atom.y) and (atom.z == prev_atom.z):
                    break
            prev_atom = atom
            keep_atoms.append(atom)
    
        new_res = Residue.from_records(keep_atoms)
        new_model.append(new_res)

    s = get_model_string([new_model])
    
    # load pose from string
    pose = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose, s)
    return pose


def mpnn_for_triads(pose, design, ligand, params=None, omit_aas='CX', num_attempts=24,
                    bias_AA_jsonl='/home/laukoa/Projects/serine_hydrolase/230404_esterase_motifs/scripts/triad_bias.jsonl'):
    print('Starting MPNN.')
    # design is string of comma separated residues be designed
    print(f'designing residues {design}')
    
    if not params:
        ligand_params = f'/home/laukoa/Projects/serine_hydrolase/theozyme/{ligand}/{ligand}.params'
    else:
        ligand_params = params

    # remove pdb info
    pose = remove_pose_matcher_lines(pose)
    chA, chB = list(pose.split_by_chain())[0], list(pose.split_by_chain())[1]

    design_selector = ResidueIndexSelector(design)

    # MPNN Design
    chA_selector = ChainSelector("A")
    mpnn_design = MPNNLigandDesign(
        design_selector=design_selector,
        params=ligand_params,
        omit_AAs=omit_aas,
        temperature=0.1,
        num_sequences=num_attempts,
        batch_size=1,
        bias_AA_jsonl=bias_AA_jsonl
    )
    # run the mpnn!
    mpnn_design.apply(pose)
    mpnn_seqs = {k: v for k, v in pose.scores.items() if "mpnn_seq" in k}

    final_pose = pyrosetta.rosetta.core.pose.Pose()
    pyrosetta.rosetta.core.pose.append_pose_to_pose(final_pose, chA, new_chain=True)
    pyrosetta.rosetta.core.pose.append_pose_to_pose(final_pose, chB, new_chain=True)

    # add the mpnn sequences to the final pose
    #for k, v in mpnn_seqs.items():
    #    pyrosetta.rosetta.core.pose.setPoseExtraScore(final_pose, k, v)
    for k, v in pose.scores.items(): #assuming `pose` had `apply` called on it
        pyrosetta.rosetta.core.pose.setPoseExtraScore(final_pose, k, v)

    # add pdb_info to final_pose
    pdb_info = pose.pdb_info()
    final_pose.pdb_info(pdb_info)

    mpnn_poses = mpnn_design.generate_all_poses(final_pose, include_native=False)
    poses = [mpnn_pose for i, mpnn_pose in enumerate(mpnn_poses)]
    return poses




