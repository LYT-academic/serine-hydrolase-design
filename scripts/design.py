#!/usr/bin/python

import sys, os, glob, json, time, shutil
import pandas as pd
import numpy as np
import argparse 
from icecream import ic
parser = argparse.ArgumentParser()

# i/o
parser.add_argument("--pdb", type=str, help="Path to pdb that needs a ligand")
parser.add_argument("--params", type=str, help="Params file. If not specified, will search in /projects/hydrolases folder.")
parser.add_argument("--enzyme_cst_f", type=str, help="Enzyme constraint file")
parser.add_argument("--odir", type=str, help="output directory")
parser.add_argument("--outprefix", type=str, default='', help="an optional prefix to add to the beginning of output pdb name")
parser.add_argument("--outsuffix", type=str, default='', help="an optional suffix to add to the beginning of output pdb name")
parser.add_argument("--startnum", type=int, default=0, help="Design number to start at")

# input key residues
parser.add_argument("--catres", type=str, help="Residue indices of the catalytic residues. Do not include oxyanion hole. (eg. if dyad, then 'ser_resi,his_resi')")
parser.add_argument("--oxhres", type=str, help="Residue indices of the oxyanion hole. Currently supporting 1 or 2 oxyanion hole residues.")
parser.add_argument("--resis_to_fix", type=str, default=None, help="Additional residues to fix in design. NOTE: if using sidechain-mediated oxyanion holes, add them here!!")
parser.add_argument("--ligand", type=str, help="Name of ligand")

# mutate residues in the structure
parser.add_argument("--mutate", type=str, default='', help="Residue indices and 1-letter residue code to mutate to. Ex: '46:H,25:NQ. If you want these fixed during design add resis_to_fix")

# design options
parser.add_argument("--load_pose_directly", action='store_true', help="Just load the pose from disk instead of pre-cleaning")
parser.add_argument("--design_relax_cycles", type=int, default=3, help="MPNN-FastRelax cycles")
parser.add_argument("--nstruct", type=int, default=1, help="number of designs to make")
parser.add_argument("--mpnn_temperature", type=float, default=0.1, help="Sampling temperature to use for MPNN")
parser.add_argument("--add_ligand", action='store_true', help="Align and add the ligand")
parser.add_argument("--initial_cartesian_relax", default=False, action='store_true', help="For the first round of relax, use cartesian relax to resolve backbone issues.")
parser.add_argument("--debug", action='store_true', help="Print some helpful shit")
parser.add_argument("--debug_rosetta", action='store_true', help="Don't mute rosetta output")
parser.add_argument("--clobber", action='store_true', default=False, help="Overwrite exsting outputs.")
parser.add_argument("--oxyanion_atom", type=str, default="O1", help="Name of oxyanion atom")
parser.add_argument("--cleanup_jsons", action='store_false', default=True, help="Don't remove json files at the end of run.")

# serine hydrolase-specific options affecting the design procedure
parser.add_argument("--fix_oxy_hole_bb", action='store_true', default=False, help="Do not fastrelax oxyanion hole backbone")
parser.add_argument("--bb_amide_oxh2", action='store_true', default=False, help="Second oxyanion contact is bb amide -- allow design")
parser.add_argument("--preorganize_histidine", action='store_true', default=False, help="Triggers a very specific sequence of events to make interactions to histidine")
parser.add_argument("--preorganize_acid", action='store_true', default=False, help="Triggers a very specific sequence of events to make interactions to catalytic acid")
parser.add_argument("--upweight_acid_near_histidine", action='store_true', default=False, help="Upweight asp/glu near the catalytic histidine")
parser.add_argument("--upweight_polars_near_oxyanion", action='store_true', default=False, help="Upweight polar residues near the oxyanion")
parser.add_argument("--upweight_nonpolars_near_ligand", action='store_true', default=False, help="Upweight nonpolar residues near the ligand (PETase)")
parser.add_argument("--downweight_activesite_arg", action='store_true', default=False, help="Downweight arginines in the active site.")
parser.add_argument("--find_hbonds_to_oxyanion", action='store_true', default=False, help="If an oxyanion hbond shows up, fix it during MPNN cycling.")
parser.add_argument("--upweight_polars_near_acid", action='store_true', default=False, help="Upweight polar residues near the acid residue")
parser.add_argument("--lg_his_constraint", action='store_true', default=False, help="Want to constrain his-lg distance (cst file must have constraint)")

# choose 1 if applicable, both are not applied simultaneously
parser.add_argument("--prohibit_proline_near_oxyanion", action='store_true', default=False, help="No prolines allowed next to the oxyanion hole.")
parser.add_argument("--force_nuc_elbow_gly", action='store_true', default=False, help="Applies to designs with nucleophilic elbows. Force GLY @ N-2 and N+2 positions.")

# options for adding ligand
parser.add_argument("--parent", type=str, help="If adding ligand, parent scaffold to take ligand from")
parser.add_argument("--parent_ser", type=int, help="If adding ligand, serine residue number of parent")

parser.add_argument("--only_score_hbonds", action='store_true', default=False, help="only check hbond filters, no design or relax")

args = parser.parse_args()
parser.set_defaults()
print("Using the following arguments:")
print(args)

import pyrosetta
import pyrosetta.distributed.io
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts

import utils
import hbond_utils
import geometry

script_path = sys.argv[0]
script_dir = os.path.abspath('/'.join(script_path.split('/')[:-1]))
repo_dir = os.path.dirname(script_dir)
mpnn_dir = f'{repo_dir}/software/LigandMPNN'

if args.params:
    params = args.params
else:
    print('Attempting to find ligand params file automatically.')
    params = f'/projects/hydrolases/serine_hydrolase/params/{args.ligand}.params'
ic(params)

enzyme_cst_f = args.enzyme_cst_f
if args.debug_rosetta:
    pyrosetta.init(f'-run:preserve_header -beta -extra_res_fa {params} -holes::dalphaball /net/software/lab/scripts/enzyme_design/DAlphaBall.gcc')
else:
    pyrosetta.init(f'-mute all -run:preserve_header -beta -extra_res_fa {params} -holes::dalphaball /net/software/lab/scripts/enzyme_design/DAlphaBall.gcc')

################ utils
from pyrosetta.rosetta.core.pose import num_chi_angles

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
aa_3_1['HIS_D'] = 'H'

acid_atoms = {'ASP': ['OD1','OD2'],
              'GLU': ['OE1','OE2']}

def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    """
    Credit: Indrek Kalvet

    Takes in a pose and calculates its SASA.
    Or calculates SASA of a given residue.
    Or calculates SASA of specified atoms in a given residue.

    Procedure by Brian Coventry
    """

    atoms = pyrosetta.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    n_ligands = 0
    for res in pose.residues:
        if res.is_ligand():
            n_ligands += 1

    for i, res in enumerate(pose.residues):
        if res.is_ligand():
            atoms.resize(i+1, res.natoms(), True)
        else:
            atoms.resize(i+1, res.natoms(), not(ignore_sc))
            if ignore_sc is True:
                for n in range(1, res.natoms()+1):
                    if res.atom_is_backbone(n) and not res.atom_is_hydrogen(n):
                        atoms[i+1][n] = True

    surf_vol = pyrosetta.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if resno is not None:
        res_surf = 0.0
        for i in range(1, pose.residue(resno).natoms()+1):
            if SASA_atoms is not None and i not in SASA_atoms:
                continue
            res_surf += surf_vol.surf(resno, i)
        return res_surf
    else:
        return surf_vol


def copy_chis(pose1, pose2, res1, res2):
    '''
    Gets the chi angles from res1 and sets the angles in res2 to the same angles.
    '''
    chis = [round(pose1.chi(i,res1),1) for i in range(1, num_chi_angles(res1,res1,pose1)+1)]
    for i, chi in enumerate(chis):
        pose2.set_chi(i+1, res2, chi)
    return pose2

def add_matcher_lines_to_pose(pose, catres, ligand, blocks=[1,2,3,4,5,6], lg_his=False, debug=False):
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
            # ligand-oxyanion hole
            line, i = make_line(i, ligand_resi, catres['oxy'], upstream_ligand=True)
            new_pdb.append(line)
            # serine-histidine
            line, i = make_line(i, catres['ser'], catres['his'], upstream_ligand=False)
            new_pdb.append(line)
            if lg_his:
                # add lg-histidine constraint
                line, i = make_line(i, ligand_resi, catres['his'], upstream_ligand=True)
                new_pdb.append(line)
            if 'acc' in catres.keys():
                # histidine-proton acceptor
                line, i = make_line(i, catres['his'], catres['acc'], upstream_ligand=False)
                new_pdb.append(line)
            if 'oxy2' in catres.keys():
                line, i = make_line(i, ligand_resi, catres['oxy2'], upstream_ligand=True)
                new_pdb.append(line)
                
        elif 'REMARK 666' in l:
            continue
        else:
            new_pdb.append(l)
    
    if debug:
        ic(catres)
        remark_lines = [line for line in new_pdb if 'REMARK' in line]
        ic(remark_lines)
    #print('\n'.join(new_pdb[:40]))
    pose2 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb))
    return pose2


def relax_with_constraints(pose, lig_resi, enzyme_cst_f, oxy_hole_resi=[], fix_oxy_hole_bb=False, oxyanion_atom="O1", cartesian=False):
    print('Starting fastrelax.')
    
    if fix_oxy_hole_bb:
        assert len(oxy_hole_resi) > 0, 'if fixing oxyanion hole backbone, must specify oxyanion hole residue indices'
        oxy_hole_resi_str = ','.join([str(x) for x in oxy_hole_resi])
        oxy_hole_selector = f'<Index name="oxy_hole" resnums="{oxy_hole_resi_str}" /> \n'
        oxy_hole_movemap = f'<ResidueSelector selector="oxy_hole" chi="1" bb="0" /> \n'

    # cartesian relax
    if cartesian:
        print("DOING CARTESIAN RELAX.")
    cart = 1 if cartesian else 0
    cart_bonded = '<Reweight scoretype="cart_bonded" weight="0.5"/>'
    pro_close = '<Reweight scoretype="pro_close" weight="0.0"/>'

    xml = f"""
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
          <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
              <Reweight scoretype="dihedral_constraint" weight="1.0"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="coordinate_constraint" weight="1.0"/>
              {cart_bonded if cartesian else ''}
              {pro_close if cartesian else ''}

          </ScoreFunction>
          <ScoreFunction name="sfxn" weights="beta"/>
      </SCOREFXNS>
      <RESIDUE_SELECTORS>
          <Chain name="chainA" chains="A"/>
          <Not name="chainB" selector="chainA"/>
          <Or name="everything" selectors="chainA,chainB"/>
          {oxy_hole_selector if fix_oxy_hole_bb else ''}
      </RESIDUE_SELECTORS>
      <TASKOPERATIONS>
          <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
          <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" ex2="0"/>
          <IncludeCurrent name="ic"/>
          <OperateOnResidueSubset name="to_relax" selector="everything"> 
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>
      </TASKOPERATIONS>
      <MOVERS>    
          <AddOrRemoveMatchCsts name="add_enz_csts" cstfile="{enzyme_cst_f}" cst_instruction="add_new"/>
          <FastDesign name="fastRelax" scorefxn="sfxn_design" repeats="1" cartesian="{cartesian}" task_operations="ex1_ex2aro,ic,limitchi2,to_relax" batch="false" ramp_down_constraints="false" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"> 
              <MoveMap name="MM" bb="1" chi="1" jump="1">
                {oxy_hole_movemap if fix_oxy_hole_bb else ''}
              </MoveMap>
          </FastDesign>
      </MOVERS>
      <FILTERS>
          <SimpleHbondsToAtomFilter name="O1_hbond" n_partners="1" hb_e_cutoff="-0.5" target_atom_name="{oxyanion_atom}" res_num="{lig_resi}" scorefxn="sfxn_design" confidence="0"/>
          <ContactMolecularSurface name="contact_molecular_surface" use_rosetta_radii="true" distance_weight="0.5" target_selector="chainA" binder_selector="chainB" confidence="0"/>
          <EnzScore name="cst_score" scorefxn="sfxn_design" confidence="0" whole_pose="1" score_type="cstE" energy_cutoff="100.0"/>
          <Ddg name="ddg"  threshold="0" jump="1" repeats="1" repack="0" confidence="0" scorefxn="sfxn"/>
      </FILTERS>
      <SIMPLE_METRICS>
      </SIMPLE_METRICS>
      <PROTOCOLS>
         <Add mover="add_enz_csts"/>
         <Add mover="fastRelax"/>
         <Add filter="O1_hbond"/>
         <Add filter="cst_score"/>
         <Add filter="contact_molecular_surface"/>
         <Add filter="ddg"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    df_scores = pd.DataFrame.from_records([packed_pose.scores])

    return pose, df_scores

def quick_pack(pose, catalytic_resis, lig_resi, enzyme_cst_f, oxy_hole_resi=[], fix_oxy_hole_bb=False):
    """
    Runs a quick fastrelax on only the sidechains of the catalytic residues and neighborhood.
    """
    print('Starting quick pack.')
    
    if fix_oxy_hole_bb:
        assert len(oxy_hole_resi) > 0, 'if fixing oxyanion hole backbone, must specify oxyanion hole residue indices'
        oxy_hole_resi_str = ','.join([str(x) for x in oxy_hole_resi])
        oxy_hole_selector = f'<Index name="oxy_hole" resnums="{oxy_hole_resi_str}" /> \n'
        oxy_hole_movemap = f'<ResidueSelector selector="oxy_hole" chi="1" bb="0" /> \n'
    
    xml = f"""
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
          <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
              <Reweight scoretype="dihedral_constraint" weight="1.0"/>
              <Reweight scoretype="angle_constraint" weight="1.0"/>
              <Reweight scoretype="coordinate_constraint" weight="1.0"/>
          </ScoreFunction>
      </SCOREFXNS>
      <RESIDUE_SELECTORS>
          <Chain name="chainA" chains="A"/>
          <Not name="chainB" selector="chainA"/>
          <Index name="catalytic" resnums="{catalytic_resis}"/>
          <Neighborhood name="catalytic_neighborhood" selector="catalytic" distance="8"/>
          <Not name="not_catalytic_neighborhood" selector="catalytic_neighborhood"/>
          {oxy_hole_selector if fix_oxy_hole_bb else ''}
      </RESIDUE_SELECTORS>
      <TASKOPERATIONS>
          <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
          <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" ex2="0"/>
          <IncludeCurrent name="ic"/>
          <OperateOnResidueSubset name="to_relax" selector="catalytic_neighborhood"> 
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="no_relax" selector="not_catalytic_neighborhood"> 
              <PreventRepackingRLT/>
          </OperateOnResidueSubset>
      </TASKOPERATIONS>
      <MOVERS>    
          <AddOrRemoveMatchCsts name="add_enz_csts" cstfile="{enzyme_cst_f}" cst_instruction="add_new"/>
          <FastDesign name="fastRelax" scorefxn="sfxn_design" repeats="1" task_operations="ex1_ex2aro,ic,limitchi2,to_relax,no_relax" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"> 
              <MoveMap name="MM" bb="0" chi="0" jump="1">
                <ResidueSelector selector="catalytic_neighborhood" chi="1" bb="1" />
                {oxy_hole_movemap if fix_oxy_hole_bb else ''}
              </MoveMap>
          </FastDesign>
      </MOVERS>
      <FILTERS>
          <EnzScore name="cst_score" scorefxn="sfxn_design" confidence="0" whole_pose="1" score_type="cstE" energy_cutoff="100.0"/>
      </FILTERS>
      <SIMPLE_METRICS>
      </SIMPLE_METRICS>
      <PROTOCOLS>
         <Add mover="add_enz_csts"/>
         <Add mover="fastRelax"/>
         <Add filter="cst_score"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    df_scores = pd.DataFrame.from_records([packed_pose.scores])

    return pose, df_scores

def add_premade_header_lines_to_pose(pose, header_lines):
    # must use -run:preserve_header !!!
    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")
    new_pdb = header_lines + pdbff
    pose2 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb))
    return pose2

def thread_seq_to_pose(pose, sequence):
    pose2 = pose.clone()
    for i, r in enumerate(sequence):
        if pose.residue(i+1).name1() == r:
            continue
        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mutres.set_target(i+1)
        mutres.set_res_name(aa_1_3[r])
        mutres.apply(pose2)
    return pose2

def get_hbset(pose):
    scorefxn = pyrosetta.get_fa_scorefxn()

    def fix_scorefxn(sfxn, allow_double_bb=False):
        opts = sfxn.energy_method_options()
        opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
        opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
        sfxn.set_energy_method_options(opts)
    fix_scorefxn(scorefxn, True)

    scorefxn(pose)
    hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
    pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbset)

    return hbset

def identify_DE_hbonds_to_catH(pose, ser_resi, his_resi):
    """
    Detect and return residue numbers for any aspartate or glutamates that
    are hydrogen bonded to the catalytic histidine.
    """

    acc, don = hbond_utils.get_resis_hbonded_to_sc(pose, his_resi, sc_only=True)

    # expect only acceptors to His ND1, but let's check for and remove
    # the catalytic serine just in case.
    acc = [res for res in acc if res != ser_resi]

    # check if any of the hbond acceptors to catalytic his are asp or glu
    acc_DE = [res for res in acc if pose.residue(res).name() in ['ASP', 'GLU']]

    # don't count cases where acceptor is one residue upstream/downstream of His (usually on helix)
    acc_DE = [res for res in acc_DE if res != his_resi + 1 or res != his_resi - 1]
    
    return acc_DE

def identify_acid_hbs(pose, acid_resis, his_resi):
    """
    Detect and return hydrogen bonding partners to any aspartate or glutamates that
    are hydrogen bonded to the catalytic histidine -- ignored histidine. The sidechain
    only hydrogen bonds are detected seperately from backbone - sidechain.

    Note: assumes the first residue in acid_resis is the acid. Don't know why this was
    all set up as lists.

    Super Important Note: We are not counting hydrogen bonds that come from Arg or Lys
    because we believe these to be non-productive interactions.

    Returns, in following order:
    1. sidechain-sidechain hbonds residue indices
    2. residue names of above residues
    3. backbone amide-sidechain hbonds residue indices
    4. residue names of above residues
    """

    if len(acid_resis) == 0:
        return [], [], [], []
    
    acid_resi = acid_resis[0] # see docstring

    # sidechain-sidechain hydrogen bonds.
    acc, don = hbond_utils.get_resis_hbonded_to_sc(pose, acid_resi, sc_only=True)
    # expect only donors and remove histidine
    sc_hb_resis = [res for res in don if res != his_resi]
    sc_hb_resns = [aa_3_1[pose.residue(res).name()[:3]] for res in sc_hb_resis]

    # exclude any interactions with lysine or arginine. those are bad.
    sc_hbs_noRK = [(res, name) 
                   for res, name in zip(sc_hb_resis, sc_hb_resns) 
                   if name not in ['R','K']]
    sc_hb_resis_noRK = [t[0] for t in sc_hbs_noRK]
    sc_hb_resns_noRK = [t[1] for t in sc_hbs_noRK]

    # backbone-sidechain hydrogen bonds.
    bb_hb_resis = hbond_utils.get_resis_hbonded_to_sc_thru_bbamide(pose, acid_resi)
    bb_hb_resns = [aa_3_1[pose.residue(res).name()[:3]] for res in bb_hb_resis]

    return sc_hb_resis_noRK, sc_hb_resns_noRK, bb_hb_resis, bb_hb_resns

def add_hbond_scores(pose, scores, hbonded_ED_resis):
    scores['num_ED_hbonds_to_his'] = [len(hbonded_ED_resis)]
    scores['ED_hbonds_to_his_resi'] = [','.join([str(x) for x in hbonded_ED_resis])]
    scores['ED_hbonds_to_his_resn'] = [','.join([pose.residue(x).name() for x in hbonded_ED_resis])]
    scores = pd.DataFrame.from_dict(scores)
    return scores
    
def save_scores(scores, outjson):
    data = {'settings':         vars(args),
            'outpdb':           outpdb,
            'outjson':          outjson,
            'scores':           scores.to_json()}
    with open(outjson, 'w') as fp:
        json.dump(data, fp)
    return True

def autoselect_acid_positions(pose, his_resi):
    sys.exit('this function has not yet been implemented. Implement it or input acid posis manually')

def mutate(pose, resi, resn):
    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
    mutres.set_target(resi)
    mutres.set_res_name(resn)
    mutres.apply(pose) 
    return pose

def bias_by_res(L, idx, AAs, fold_increase):
    '''
    Author: Dr. Doug Tischer

    Inputs
    -----------------
    idx (list of ints): idx1! (To match the rest of mpnn indexing)
    AAs (list of str): Single AA letter of amino acid to bias. ex: ['SDE', 'IVL', 'KR']
    fold_increase (list of floats): How much to increase likelyhood of AA being choosen.
    
    idx, aa and fold_increase must be the same length 
    
    Outputs
    -----------------
    bias (np.array, (L, 21)): log of the fold increase at each position. This bias will be added to the predicted mpnn logits.
    '''
    alphabet_mpnn = list('ACDEFGHIKLMNPQRSTVWYX')
    fold = np.ones((L, 21))
    
    for i, aas, f in zip(idx, AAs, fold_increase):
      for aa in aas:
        fold[i-1, alphabet_mpnn.index(aa)] = f
      
    bias = np.log(fold)
    return bias

def atom_neighborhood(pose, resi, atom, distance=7.0):
    """
    Returns a list of the indices of residues whose CAs are within 
    some distance of atom in residue resi.
    """
    neighbor_residues = []
    atom_xyz = pose.residue(resi).xyz(atom)

    for res in pose.residues:
        if res.seqpos() == resi:
            continue
        elif res.is_ligand():
            continue
        elif (atom_xyz - res.xyz('CA')).norm() <= distance:
            neighbor_residues.append(res.seqpos())
   
    return neighbor_residues 

def directional_atom_neighborhood(pose, resi, atom, anchor_resi, anchor_atom, distance=7.0):
    """
    Returns the indices of residues whose CAs are within some distance 
    of atom in residue resi, AND whose CAs are in the direction of the
    vector defined by anchor_atom -> atom.
    """
    neighbor_residues = []
    
    # get the reference vector (anchor_atom->atom)
    atom_xyz = np.array(pose.residue(resi).xyz(atom))
    anchor_atom_xyz = np.array(pose.residue(resi).xyz(anchor_atom))
    ref_vec = atom_xyz - anchor_atom_xyz    

    for res in pose.residues:
        if res.seqpos() == resi:
            continue
        if res.is_ligand():
            continue

        CA_xyz = np.array(res.xyz('CA'))
       
        if np.sqrt(np.sum((atom_xyz - CA_xyz) ** 2, axis=0)) <= distance:
            # get the vector between atom and this residue's CA (atom->CA)
            vec = CA_xyz - atom_xyz

            # Math: if the dot product of two vectors is positive, they
            # form an acute angle (aka. point in the same direction).

            if np.dot(ref_vec, vec) > 0:
                neighbor_residues.append(res.seqpos())
   
    return neighbor_residues 

def measure_Nbb_oxyanion_hole(pose, resi, oxyanion='O1'):
    """
    Measure the distance, angle, and dihedralB for oxyanion-backbone 
    interaction. Atoms in order: oxyanion atom (ligand), oxyanion 
    residue N, preceding residue Ca, O.
    """
    ligand = pose.residue(pose.size())
    if not ligand.is_ligand():
        sys.exit("Can't find the ligand in pose")
    Ox = np.array(ligand.xyz(oxyanion))
    N  = np.array(pose.residue(resi).xyz('N'))
    C  = np.array(pose.residue(resi-1).xyz('C'))
    O  = np.array(pose.residue(resi-1).xyz('O'))
    
    distance = geometry.measure_distance(Ox, N)
    angle = geometry.measure_angle(Ox, N, C)
    dihedral = geometry.measure_dihedral(Ox, N, C, O)

    measurements = {'distance': round(distance, 2),
                    'angle': round(angle, 1),
                    'dihedral': round(dihedral, 1)}
    
    return measurements

def is_Nbb_oxyanion_hole(pose, resi, oxyanion='O1'):
    """
    Measure the oxyanion hole--oxyanion interaction, decide whether
    or not the pair is hydrogen bonded based on extremely scientific
    cutoffs set by AL. Uses absolute value of dihedral angle. There
    must be an oxyanion-containing ligand in the structure.
    """
    cutoffs = {'distance': (2.4, 3.2),
               'angle':    (100, 150),
               'dihedral': (135, 180)}
    geom = measure_Nbb_oxyanion_hole(pose, resi, oxyanion=oxyanion)
    for k in geom:
        v = geom[k]
        lower, upper = cutoffs[k][0], cutoffs[k][1]
        if not (abs(v) >= lower and abs(v) <= upper):
            return False
    return True

def score_oxyanion_hole(pose, scores={}, oxh_resi=[], oxyanion_atom='O1'):
    # inputs are oxyanion atom, oxh resi, scores object, pose
    # first make measurements of fixed bb amide oxyanion hole interactions
    for i, resi in enumerate(oxh_resi):
        scores[f'is_oxh{i+1}_hbond'] = [int(is_Nbb_oxyanion_hole(pose, resi, oxyanion=oxyanion_atom))]

    # then look for serendipitous hbonds from other residues
    don, acc = hbond_utils.get_resis_hbonded_to_atom(pose, oxyanion_atom, pose.size())
    don_atms, acc_atms = hbond_utils.get_atoms_hbonded_to_atom(pose, oxyanion_atom, pose.size())
    # categorize the serendipitous hbonds into sidechain or backbone mediated
    add_sc_oxhb = [] # [(resi, resn, atom)]
    add_bb_oxhb = [] # [(resi, resn, atom)]
    for atom, resi in zip(don_atms, don):
        resn = pose.residue(resi).name()
        if atom == 'H': # that's the bb amide H.
            add_bb_oxhb.append((resi, resn, atom))
        else:
            add_sc_oxhb.append((resi, resn, atom))

    # add the additional O1 hbond info to scores.
    scores['bb_Ox_hbond_resi'] = [','.join([str(n[0]) for n in add_bb_oxhb])]
    scores['bb_Ox_hbond_resn'] = [','.join([str(n[1]) for n in add_bb_oxhb])]
    scores['bb_Ox_hbond_atms'] = [','.join([str(n[2]) for n in add_bb_oxhb])]

    scores['sc_Ox_hbond_resi'] = [','.join([str(n[0]) for n in add_sc_oxhb])]
    scores['sc_Ox_hbond_resn'] = [','.join([str(n[1]) for n in add_sc_oxhb])]
    scores['sc_Ox_hbond_atms'] = [','.join([str(n[2]) for n in add_sc_oxhb])]

    return scores

####################################################################################

outprefix = ''
if len(args.outprefix) > 0:
    outprefix = args.outprefix + '_'

for n in range(args.startnum, args.startnum+args.nstruct):
    t0 = time.time()
    
    inpdb_name = args.pdb.split('/')[-1].replace('.pdb','')
    # replace any dumb diffusion output suffixes.
    inpdb_name = inpdb_name.replace('-atomized-bb-False','')
    inpdb_name = inpdb_name.replace('-atomized-bb-True', '')

    outpdb = f"{args.odir}/{outprefix}{inpdb_name}_{args.ligand}_{n}{args.outsuffix}.pdb"

    # check for existing outputs
    if not args.clobber:
        if not args.only_score_hbonds and os.path.exists(outpdb):
            print('found existing output, skipping.')
            continue
    os.makedirs(args.odir, exist_ok=True)

    # load the diffusion pose with a special function that ignores overlapping sidechain atoms.
    if args.load_pose_directly:
        pose = pyrosetta.pose_from_file(args.pdb)
    else:
        pose = utils.load_diffusion_pose(args.pdb)
    
    # make mutations
    if args.mutate != '':
        # parse argument
        mutate_strs = args.mutate.split(',')
        mutations = [(int(a.split(':')[0]), a.split(':')[1]) for a in mutate_strs]
        ic(mutations)
        # make mutations
        for mutation in mutations:
            resi = mutation[0]
            resn = aa_1_3[mutation[1]]

            if pose.residue(resi).name3() == resn:
                print(f'Position {resi} is already {resn}. Skipping mutation.')
                continue

            print(f'Mutating position {resi} to {resn}.')
            pose = mutate(pose, resi, resn)

    # add in the ligand
    if args.add_ligand:
        if not args.parent:
            sys.exit('need a parent scaffold to steal the ligand from!')
        if not args.parent_ser:
            sys.exit('need the parent serine resi to align to')

        parent = pyrosetta.pose_from_file(f'{args.parent}')
        parent_resi = [args.parent_ser]
        pose_resi = [int(args.catres.split(',')[0])]

        # align the pdb with original on the serine backbone atoms, copy in ligand
        pose = utils.align_n_add_ligand(pose, parent, pose_resi, parent_resi)

    ###################################################################
    # build matcher header for enzyme constraints                     #
    ###################################################################

    if args.lg_his_constraint:
        print('adding lg-his constraint')
   
    # parsing the catalytic residues (residue idxs of the dyad/triad network)
    cat_resi = [int(x) for x in args.catres.split(',')]
    assert len(cat_resi) in [2,3], 'found an unexpected number of catalytic residues'
    # dyad
    if len(cat_resi) == 2:
        catres = {'ser': cat_resi[0], 'his': cat_resi[1]}
    # triad
    elif len(cat_resi) == 3:
        catres = {'ser': cat_resi[0], 'his': cat_resi[1], 'acc': cat_resi[2]}
       
    # parsing the oxyanion hole residues
    oxh_resi = [int(x) for x in args.oxhres.split(',')]
    assert len(oxh_resi) in [1,2], 'found an unexpected number of oxyanion hole residues'
    catres['oxy'] = oxh_resi[0]
    if len(oxh_resi) == 2:
        catres['oxy2'] = oxh_resi[1]

    # add the constraint file lines to the pose
    pose = add_matcher_lines_to_pose(pose, catres, args.ligand, lg_his=args.lg_his_constraint, debug=args.debug)
                
    ser_resi = catres['ser']
    his_resi = catres['his']
    
    print(f'Using cst file {enzyme_cst_f}')

    ###################################################################
    # run design                                                      #
    ###################################################################

    # score the hbonds on the input and quit
    if args.only_score_hbonds:
        outjson = f"{args.odir}/{outprefix}{inpdb_name}_hbonds.json"
        hbonded_ED_resis = identify_DE_hbonds_to_catH(pose, ser_resi=ser_resi, his_resi=his_resi)
        scores = add_hbond_scores(pose, {}, hbonded_ED_resis)
        scores = score_oxyanion_hole(pose, scores, oxh_resi, args.oxyanion_atom)
        # hbonds to acid
        acid_sc_hb_resis, acid_sc_hb_resns, acid_bb_hb_resis, acid_bb_hb_resns = \
                            identify_acid_hbs(pose, acid_resis=hbonded_ED_resis, his_resi=his_resi)
        scores['acid_sc_hb_resis'] = ','.join([str(k) for k in acid_sc_hb_resis])
        scores['acid_sc_hb_resns'] = ','.join(acid_sc_hb_resns)
        scores['acid_bb_hb_resis'] = ','.join([str(k) for k in acid_bb_hb_resis])
        scores['acid_bb_hb_resns'] = ','.join(acid_bb_hb_resns)

        save_scores(scores, outjson)
        sys.exit('Finished scoring hbonds.')

    # fixed the backbone of the oxyanion hole, so that it doesn't
    # minimize out of the desired orientation
    if args.fix_oxy_hole_bb or args.prohibit_proline_near_oxyanion:
        if not args.bb_amide_oxh2: # second oxy contact is sc -- not to be fixed, only with N motifs
            oxy_hole_resis = [catres['ser'], catres['oxy']]
            oxy_hole_resis = list(set(oxy_hole_resis)) # drop duplicates
            print(f'fixing {oxy_hole_resis}')
        else: # second oxy contact is bb amide -- fix, corresponds with N+1 motifs
            oxy_hole_resis = [catres['oxy'], catres['oxy2']]
            oxy_hole_resis = list(set(oxy_hole_resis)) # drop duplicates
            print(f'fixing {oxy_hole_resis}')

    
    # write the omit AA json to omit amino acids in MPNN per position.
    # note, the below path just doesn't get written if argument not included
    omitAA_json = f"{args.odir}/{outprefix}{inpdb_name}_{n}_omitAA.jsonl"
    omitAA_dict = {"tmp": {"A": []}}

    if args.prohibit_proline_near_oxyanion:
        # this only really applies to N oxyanion holes.
        if ser_resi == oxy_hole_resis[0]:
            print('Disallowing proline at the N-1 position.')
            omitAA_dict["tmp"]["A"].append([[ser_resi-1], "P"])
            # if not os.path.exists(omitAA_json):
            #     with open(omitAA_json, 'w') as f:
            #         # disallow proline at N+1
            #         json.dump({"tmp":{"A": [[[ser_resi-1], "P"]]}}, f)
        print('Disallowing proline at the oxyanion hole positions')
        print(oxy_hole_resis[0], oxy_hole_resis[1])
        omitAA_dict["tmp"]["A"].append([[oxy_hole_resis[0]], "P"])
        omitAA_dict["tmp"]["A"].append([[oxy_hole_resis[1]], "P"])

    elif args.force_nuc_elbow_gly:
        # only applies if there is a nucleophilic elbow in design.
        omitAA_dict["tmp"]["A"].append([[ser_resi-2,ser_resi+2], "ARNDCEQHILKMFPSTWYV"])
        # if not os.path.exists(omitAA_json):
        #     with open(omitAA_json, 'w') as f:
        #         # disallow all aa's except G at N-2 and N+2
        #         json.dump({"tmp":{"A": [[[ser_resi-2,ser_resi+2], "ARNDCEQHILKMFPSTWYV"]]}}, f)

    if args.downweight_activesite_arg:
        # define the active site neighborhood
        activesite_neighborhood = []
        for res in cat_resi+oxh_resi:
            neighbors = atom_neighborhood(pose, resi=res, atom='CA', distance=10.0)
            activesite_neighborhood += neighbors
        activesite_neighborhood = [*set(activesite_neighborhood)]
        print(f'disallowing arginine at the following positons: {"+".join([str(x) for x in activesite_neighborhood])}')

        # omit Arg at the neighborhood positions
        # ic(activesite_neighborhood)
        omitAA_dict["tmp"]["A"].append([activesite_neighborhood, "R"])
    # ic(omitAA_dict)
    # write the omitAA_dict if there's anything in there.
    if len(omitAA_dict["tmp"]["A"]) == 0:
        with open(omitAA_json, 'w') as f:
            json.dump(omitAA_dict, f)

    # cycle MPNN and FastRelax (with constraints)
    mpnn_design_res = [res for res in range(1, pose.size()) if res not in cat_resi]

    # remove any additional fixed residues from the designable set
    if args.resis_to_fix:
        resis_to_fix = [int(x) for x in args.resis_to_fix.split(',')]
        mpnn_design_res = [res for res in mpnn_design_res if res not in resis_to_fix]

    # if present, set second oxyanion hole to not designable if sc
    # if 2nd oxh contact is bb amide, allow design
    if len(oxh_resi) == 2 and not args.bb_amide_oxh2:
    #if len(oxh_resi) == 2:
        if oxh_resi[1] in mpnn_design_res:
            mpnn_design_res.remove(oxh_resi[1])

    # remove any already hbonded residues from designable set
    hbonded_ED_resis = identify_DE_hbonds_to_catH(pose, ser_resi=ser_resi, his_resi=his_resi)
    mpnn_design_res = [res for res in mpnn_design_res if res not in hbonded_ED_resis]
    if len(hbonded_ED_resis) > 0:
        print(f'Detected hbonded E/D {hbonded_ED_resis} and fixed them')
    # fix good hbonds to catalytic acid
    acid_sc_hb_resis, acid_sc_hb_resns, acid_bb_hb_resis, acid_bb_hb_resns = \
                            identify_acid_hbs(pose, acid_resis=hbonded_ED_resis, his_resi=his_resi)
    mpnn_design_res = [res for res in mpnn_design_res if res not in acid_sc_hb_resis]
    if len(hbonded_ED_resis) > 0:
        print(f'Detected hbonding resis to catalytic acid: {acid_sc_hb_resis} {acid_sc_hb_resns} and fixed them')

    #############################################################################
    # start MPNN fastrelax cycling
    #############################################################################

    for i in range(args.design_relax_cycles):

        # adding bias on a per-residue basis, if applicable
        bias_idx = []
        bias_aas = []
        fold_increase = []


        if i == args.design_relax_cycles - 1:
            print('--> Entering final design/relax cycle! <--')

        # upweighting acid (asp/glu) residues near the catalytic histidine
        if args.upweight_acid_near_histidine:

            if i != args.design_relax_cycles - 1: # do not apply bias on final cycle.
                # NOTE: if there is already an asp | glu hbonded to the his, this process is aborted!
                if len(hbonded_ED_resis) > 0:
                    print(f'Hbonded Asp & Glu to catalytic His already present.')

                else:
                    print('Upweighting Asp & Glu near the catalytic histidine')
                    # select a 'hemisphere' of residues that could potentially Hbond to histidine-ND1
                    # note: we select directionally
                    nd1_neighbors = directional_atom_neighborhood(pose, resi=his_resi, atom='ND1',
                                                                  anchor_resi=his_resi, anchor_atom='NE2')
                    ic(nd1_neighbors)
                    
                    bias_idx += nd1_neighbors
                    bias_aas += ['ED'] * len(nd1_neighbors)
                    fold_increase += [10] * len(nd1_neighbors)
        
        # upweight polar residues seen near oxyanion hole in natives
        if args.upweight_polars_near_oxyanion:
            
            if i != args.design_relax_cycles - 1: # do not apply this bias on final cycle.
                print('Upweighting polars near the oxyanion')
                # select a shell around the oxyanion (assume ligand is last in sequence)
                oxyanion_neighbors = atom_neighborhood(pose, resi=pose.size(), atom=args.oxyanion_atom)
                ic(oxyanion_neighbors)

                bias_idx += oxyanion_neighbors
                bias_aas += ['STYWNQ'] * len(oxyanion_neighbors) 
                fold_increase += [10] * len(oxyanion_neighbors) 

        if args.upweight_polars_near_acid:
            print('Upweighting polars near the acid residue.')
            # check for acid residue hbonded to histidine
            if len(hbonded_ED_resis) == 0:
                print('No residues detected hbonded to histidine, skipping.')
            else:
                hbonded_ED_resns = [pose.residue(x).name() for x in hbonded_ED_resis]
                # not great, but just considering the first hbonded residue in list as the acid
                acid_resi = hbonded_ED_resis[0]
                acid_resn = hbonded_ED_resns[0]

                # select shell around both chemically relevant acid atoms
                atoms = acid_atoms[acid_resn]
                acid_neighbors = []
                for atom in atoms:
                    acid_neighbors += atom_neighborhood(pose, resi=acid_resi, atom=atom)
                    acid_neighbors = [*set(acid_neighbors)]

                ic(acid_neighbors)
                bias_idx += acid_neighbors
                bias_aas += ['STYHWNQ'] * len(acid_neighbors)
                fold_increase += [5] * len(acid_neighbors) 

        if args.upweight_nonpolars_near_ligand:
            print('Upweighting nonpolars at the ligand binding site')
            # define the active site neighborhood
            activesite_neighborhood = []
            for res in cat_resi+oxh_resi:
                neighbors = atom_neighborhood(pose, resi=res, atom='CA', distance=10.0)
                activesite_neighborhood += neighbors
            activesite_neighborhood = [*set(activesite_neighborhood)]

            ic(activesite_neighborhood)
            bias_idx += activesite_neighborhood
            bias_aas += ['FILMVWY'] * len(activesite_neighborhood)
            fold_increase += [10] * len(activesite_neighborhood)

        # make array of per residue amino acid bias
        bias = bias_by_res(L=pose.size()-1, idx=bias_idx,
                           AAs=bias_aas, fold_increase=fold_increase)
        bias_json = f"{args.odir}/{outprefix}{inpdb_name}_{n}_per_res_bias.jsonl"
        with open(bias_json, 'w') as f:
            json.dump({"tmp":{"A": bias.tolist()}}, f)

        # dump pdb for mpnn
        tmp_pdb = f'{args.odir}/TMP_{outprefix}{inpdb_name}_{args.ligand}_{n}{args.outsuffix}.pdb'
        pose.dump_pdb(tmp_pdb)

        # run mpnn
        redesigned_residues = ' '.join([f'A{i}' for i in mpnn_design_res])
        mpnn_cmd = f"""python {mpnn_dir}/run.py \
                              --pdb_path {tmp_pdb} \
                              --out_folder {args.odir} \
                              --model_type ligand_mpnn \
                              --checkpoint_ligand_mpnn ../../software/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
                              --redesigned_residues "{redesigned_residues}" \
                              --bias_AA_per_residue {bias_json} \
                              --temperature {args.mpnn_temperature}"""
        if os.path.exists(omitAA_json):
            mpnn_cmd += ' --omit_AA_per_residue {omitAA_json} '
        os.system(mpnn_cmd+'\n')
        pose = pyrosetta.pose_from_file(f'{args.odir}/backbones/TMP_{outprefix}{inpdb_name}_{args.ligand}_{n}{args.outsuffix}_1.pdb')

        # fastrelax with enzyme constraints
        pose = add_matcher_lines_to_pose(pose, catres, args.ligand, lg_his=args.lg_his_constraint)
        cartesian_relax = True if args.initial_cartesian_relax and i == 0 else False
        pose, scores = relax_with_constraints(pose, 
                                              lig_resi=pose.size(), 
                                              enzyme_cst_f=enzyme_cst_f, 
                                              fix_oxy_hole_bb=args.fix_oxy_hole_bb, 
                                              oxy_hole_resi=oxy_hole_resis,
                                              oxyanion_atom=args.oxyanion_atom,
                                              cartesian=cartesian_relax)

        # score the acid sasa (if exists)
        if len(hbonded_ED_resis) == 0:
            scores['acid_sasa'] = 0.0
        else:
            scores['acid_sasa'] = getSASA(pose, resno=hbonded_ED_resis[0])
        
        # score the oxyanion hydrogen bonds
        scores = score_oxyanion_hole(pose, scores, oxh_resi, args.oxyanion_atom)

        # if any Ox hbonds come from sidechains, fix those residue identities for next round.
        if args.find_hbonds_to_oxyanion:
            # find hbonds to oxyanion hole from sidechains
            if scores['sc_Ox_hbond_resi'][0] != '':
                sc_Ox_hbond_resi = [int(x) for x in scores['sc_Ox_hbond_resi'][0].split(',')]
                mpnn_design_res = [res for res in mpnn_design_res if res not in sc_Ox_hbond_resi]
                print(f'Detected hbond(s) to {args.oxyanion_atom} from {sc_Ox_hbond_resi} and fixed.')

        # if preorganizing histidine, find key hbonds to his and fix them for the next mpnn round!
        if args.preorganize_histidine:
            # detect and fix any E/D hbonded to catalytic H
            hbonded_ED_resis = identify_DE_hbonds_to_catH(pose, ser_resi=ser_resi, his_resi=his_resi)
            mpnn_design_res = [res for res in mpnn_design_res if res not in hbonded_ED_resis]
            scores = add_hbond_scores(pose, scores, hbonded_ED_resis)
            if len(hbonded_ED_resis) > 0:
                print(f'Detected hbonded E/D {hbonded_ED_resis} and fixed them')
            #TODO: it would improve the process to check these fixed residues each time and make
            # sure the oxyanion hbond hasn't been lost in subsequent rounds (if it has, add the 
            # previously fixed residue back into the designable set.

        # identify hydrogen bonds to the acid
        acid_sc_hb_resis, acid_sc_hb_resns, acid_bb_hb_resis, acid_bb_hb_resns = \
                            identify_acid_hbs(pose, acid_resis=hbonded_ED_resis, his_resi=his_resi)
        scores['acid_sc_hb_resis'] = ','.join([str(k) for k in acid_sc_hb_resis])
        scores['acid_sc_hb_resns'] = ','.join(acid_sc_hb_resns)
        scores['acid_bb_hb_resis'] = ','.join([str(k) for k in acid_bb_hb_resis])
        scores['acid_bb_hb_resns'] = ','.join(acid_bb_hb_resns)

        # if preorganizing acid, fix any sidechains that are hydrogen bonding with the acid
        if args.preorganize_acid:
            if len(acid_sc_hb_resis) > 0:
                print(f'Detected sidechain hbonds to acid from {acid_sc_hb_resis} and fixed them')
            mpnn_design_res = [res for res in mpnn_design_res if res not in acid_sc_hb_resis]

    ###################################################################
    # save results                                                    #
    ###################################################################

    pose.dump_pdb(outpdb)
    outjson = f"{args.odir}/{outprefix}{inpdb_name}_{args.ligand}_{n}{args.outsuffix}.json"
    save_scores(scores, outjson)

    # clean up json files
    if args.cleanup_jsons:
        files_to_delete = [omitAA_json, bias_json, tmp_pdb]
        dirs_to_delete = [f'{args.odir}/backbones', f'{args.odir}/seqs']
        for f in files_to_delete:
            if os.path.exists(f):
                os.remove(f)
        for d in dirs_to_delete:
            if os.path.exists(d):
                shutil.rmtree(d)

    t1 = time.time()
    print(f'Finished making design number {n} in {round(t1-t0)} seconds')

