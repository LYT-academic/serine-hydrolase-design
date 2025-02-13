
import sys, glob, os
from Prediction_v3 import Prediction

import pandas as pd
import numpy as np
from icecream import ic


alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}


# dictionary to keep track of atom pairs for rmsd calculations
RMSD_TO_REF = {
    'bu2': {'tet1':[['C14','C13','C10', 'C1', 'O1', 'O2', 'C2','C11', 'C9', 'O3', 'C8', 'O4', 'C7', 'C6','C12', 'C5', 'C4', 'C3'], 
                    ['C31','C30','C27','C1A','OAC','O17','C16','C28','C26','O19','C25','O21','C24','C23','C29','C22','C20','C18']],
            'tet2':[['C14','C13','C10', 'C1', 'O1', 'O2'],
                    ['C20','C19','C18','C1A','OAC','O17']]},
    # this substrate's acyl group is symmetric, so just calculate rmsd for the two carbons down chain & end carbon
    'bn2': {'tet1':[['C16','C13','C10', 'C1', 'O1', 'O2', 'C2','C11', 'C9', 'O3', 'C8', 'O4', 'C7', 'C6','C12', 'C5', 'C4', 'C3'], 
                    ['C33','C30','C27','C1A','OAC','O17','C16','C28','C26','O29','C25','O21','C24','C23','C29','C22','C20','C18']],
            'tet2':[['C16','C13','C10', 'C1', 'O1', 'O2'],
                    ['C21','C18','C17','C1A','OAC','O16']]},
    'mu2': {'tet1':[['C10', 'C1', 'O1', 'O2', 'C2', 'C3', 'C4', 'C5', 'C6','C12', 'C7', 'C8', 'O4', 'O3', 'C9','C11'], 
                    ['C27','C1A','OAC','O17','C16','C18','C20','C22','C23','C29','C24','C25','O21','O19','C26','C28']],
            'tet2':[['C10', 'C1', 'O1', 'O2'],
                    ['C18','C1A','OAC','O17']]},
    'mu1': {'tet1':[['C10', 'C1', 'O1', 'O2', 'C2', 'C3', 'C4', 'C5', 'C6','C12', 'C7', 'C8', 'O4', 'O3', 'C9','C11'], 
                    ['C27','C1A','OAC','O17','C16','C18','C20','C22','C23','C29','C24','C25','O21','O19','C26','C28']],
            'tet2':[['C10', 'C1', 'O1', 'O2'],
                    ['C18','C1A','OAC','O17']]},
}

LG_NAMES = {
    'bu2': {'tet1': 'O17', 'tet2': 'O17'},
    'bn2': {'tet1': 'O17', 'tet2': 'O16'},
    'mu2': {'tet1': 'O17', 'tet2': 'O17'},
    'mu1': {'tet1': 'O17', 'tet2': 'O17'}
}

# dictionary of 3-atom lists to look for hbonds through for oxyanion hole
OXH_ATOMS = {
    'bb': [['N','CA','C']],
    'H' : [['NE2','CE1','ND1'], ['ND1','CE1','NE2']],
    'S' : [['OG','CB','CA']],
    'T' : [['OG1','CB','CA']],
    'Y' : [['OH','CZ','CE1']],
    'W' : [['NE1','CE2','CD2']],
    'N' : [['ND2','CG','OD1']],
    'Q' : [['NE2','CD','OE1']],
    'R' : [['NH2','CZ','NE'], ['NE','CZ','NH2'], ['NH1','CZ','NE']],
    'K' : [['NZ','CE','CD']]
}

# hybridization of hbonding atoms
ATM_HYBRID = {
    'sp2': ['bb','NE2','ND1','NE1','ND2','NE2','NH2','NE','NH1',
            'OD1','OD2','OE1','OE2','N'],
    'sp3': ['OG','OG1','OH','NZ','OAC','O2','O1','O17','SG']
}

# atoms with 'planar' environment (dihedral should be near 0/180)
# NE2 sucks b/c it's in His and Gln but has different possible dihedrals.
ATMS_PLANAR_0_180 = ['ND2','NH2','NH1','OD1','OD2','OE1','OE2','NE2']
ATMS_PLANAR_180   = ['ND1','NE1','NE']

# H bond geometry cutoffs by hybridization
HB_CUTOFFS = {
    'sp2': {'distance':   (2.0,    3.6),
            'angleA'  :   (80.0,  160.0),
            'angleB'  :   (80.0,  160.0),
            'dihedralA':  None,
            'dihedralAB': None,
            'dihedralB':  None},

    'sp3': {'distance':   (2.0,    3.6),
            'angleA'  :   (69.5, 149.5),
            'angleB'  :   (69.5, 149.5),
            'dihedralA':  None,
            'dihedralAB': None,
            'dihedralB':  None},
}

def make_metric_dict(ser_idx, his_idx, asp_idx, ox1_idx, ox2_idx, prot_chain, lig_idx, 
                     lig_chain, step, oxhresns=[], atomic_info=None, oxyanion_atom=None,
                     acylMe_atom=None, acylattack_atom=None, ligand_name=None, nuc='ser',
                     his_acc='NE2', his_don='ND1'):
    print("STEP", step)
       
    oxhres = [idx for idx in [ox1_idx, ox2_idx] if idx]
    # ic(oxhres)

    if nuc == 'ser':
        nucatm = 'OG'
        refatm = 'OG'
    elif nuc == 'cys':
        refatm = 'SG'
        # currently everythings set up to make oxygen intermediates. sorry...
        if step in ['free','acyl']:
            nucatm = 'SG'
        else:
            nucatm = 'OG'

    # Hard-coded dictionary of inter-residue interactions
    ic(prot_chain)
    ic(ser_idx)
    ic(nucatm)
    metrics2measure = {
        'distance': {
            'serhis_hbdist':
                {'name': 'serhis_hbdist',
                 'atm_maps': [{'chain': prot_chain, 'resnum': ser_idx, 'atom_name':  nucatm, 'ref': False},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc, 'ref': False}],
                 'compare2ref': False}, # was True
            'hislg_hbdist_toreflig':
                {'name': 'hislg_hbdist_toreflig',
                 'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc, 'ref': False},
                              {'chain':  lig_chain, 'resnum': lig_idx, 'atom_name':    'O2',  'ref': True}],
                 'compare2ref': False},

        }, 
        'angle'   : {
            # Ser-His angleA
            'serhis_angleA':
                {'name': 'serhis_angleA',
                 'atm_maps': [{'chain': prot_chain, 'resnum': ser_idx, 'atom_name':    'CB'},
                              {'chain': prot_chain, 'resnum': ser_idx, 'atom_name':  nucatm},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc}],
                 'compare2ref': False}, # was True
            # Ser-His angleB
            'serhis_angleB':
                {'name': 'serhis_angleB',
                 'atm_maps': [{'chain': prot_chain, 'resnum': ser_idx, 'atom_name': nucatm},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name': 'CE1'}],
                 'compare2ref': False}, # was True
            # His - LG angles (to reference ligand)
            'hislg_angleB_toreflig':
                {'name': 'hislg_angleB_toreflig',
                 'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc, 'ref': False},
                              {'chain':  lig_chain, 'resnum': lig_idx, 'atom_name': 'O2',  'ref': True},
                              {'chain':  lig_chain, 'resnum': lig_idx, 'atom_name': 'C1',  'ref': True}],
                 'compare2ref': False},
            'hislg_angleA_toreflig':
                {'name': 'hislg_angleA_toreflig',
                 'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name':   'CE1', 'ref': False},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc, 'ref': False},
                              {'chain':  lig_chain, 'resnum': lig_idx, 'atom_name': 'O2',  'ref': True}],
                 'compare2ref': False},
        },
        'dihedral': {
            # Serine sidechain chi1
            'ser_chi':
                {'name': 'ser_chi',
                 'atm_maps': [{'chain': prot_chain, 'resnum': ser_idx, 'atom_name': atom} 
                               for atom in ['N','CA','CB',nucatm]],
                 'compare2ref': True}, # was True
            # Ser - His hbond dihedral
            'serhis_dihedralB':
                {'name': 'serhis_dihedralB',
                 'atm_maps': [{'chain': prot_chain, 'resnum': ser_idx, 'atom_name':  nucatm},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name':   'CE1'},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_don}],
                 'compare2ref': False}, # was True
            # His - LG hbond dihedral (to reference ligand)
            'hislg_dihedralA_toreflig':
                {'name': 'hislg_dihedralA_toreflig',
                 'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_don, 'ref': False},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name':   'CE1', 'ref': False},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc, 'ref': False},
                              {'chain':  lig_chain, 'resnum': lig_idx, 'atom_name': 'O2',  'ref': True}],
                 'compare2ref': False}, # was True

        },
        'rmsd'    : {
            'ser_sc_rmsd_toref':
                {'name': 'ser_sc_rmsd_toref',
                 'atm_maps1': [{'chain': prot_chain, 'resnum': ser_idx, 'atom_name': atom}
                                for atom in ['CA','CB',nucatm]],
                 'atm_maps2': [{'chain': prot_chain, 'resnum': ser_idx, 'atom_name': atom, 'ref': True}
                                for atom in ['CA','CB',refatm]]}, 
            'his_sc_rmsd_toref':
                {'name': 'his_sc_rmsd_toref',
                 'atm_maps1': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name': atom}
                                for atom in ['CB','CG','ND1','CE1','NE2','CD2']],
                 'atm_maps2': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name': atom, 'ref': True}
                                for atom in ['CB','CG','ND1','CE1','NE2','CD2']]}, 
        },

        'uncerts' : {
            'serOG_unc':
                {'name': 'serOG_unc',  
                 'atm_map': {'chain': prot_chain, 'resnum': ser_idx, 'atom_name': nucatm}},
            'hisNE2_unc':
                {'name': 'hisNE2_unc', 
                 'atm_map': {'chain': prot_chain, 'resnum': his_idx, 'atom_name': 'NE2'}},
            'hisND1_unc':
                {'name': 'hisND1_unc', 
                 'atm_map': {'chain': prot_chain, 'resnum': his_idx, 'atom_name': 'ND1'}},
        }
    }

    if asp_idx:

        # get the identity of the acid from atomic info object
        acid_atoms = atomic_info[prot_chain][asp_idx]
        if 'OD1' in acid_atoms:
            atoms = 'OD1,OD2'
        elif 'OE1' in acid_atoms:
            atoms = 'OE1,OE2'
        acid_sc_atoms = [atom for atom in acid_atoms if atom not in ['N','CA','C','O']]
    
        # his-"acid" hbond
        metrics2measure['distance']['hisacid_hbdist'] = \
            {'name': 'hisacid_hbdist',
             'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_don, 'ref': False},
                          {'chain': prot_chain, 'resnum': asp_idx, 'atom_name': atoms, 'ref': False}],
             'compare2ref': False}

        # "acid" uncertainty
        metrics2measure['uncerts']['acidOXY_unc'] = \
            {'name': 'acidOXY_unc', 
             'atm_map': {'chain': prot_chain, 'resnum': asp_idx, 'atom_name': atoms}}

        # "acid" sidechain rmsd
        metrics2measure['rmsd']['acid_sc_rmsd_toref'] = \
                {'name': 'acid_sc_rmsd_toref',
                 'atm_maps1': [{'chain': prot_chain, 'resnum': asp_idx, 'atom_name': atom}
                                for atom in acid_sc_atoms],
                 'atm_maps2': [{'chain': prot_chain, 'resnum': asp_idx, 'atom_name': atom, 'ref': True}
                                for atom in acid_sc_atoms]}
   
    for k, oxh in enumerate(oxhres):
        ic(oxhresn)
        
        # get interacting atom for this oxyanion hole residue
        resn = oxhresns[k]
        oxhatom = ','.join([atmset[0] for atmset in OXH_ATOMS[resn]])
        
        # calculate oxyanion hole sidechain rmsd if applicable
        if resn == 'sc':
            
            # get the oxyanion atoms (sidechain) for rmsd calculation
            ox_atoms = atomic_info[prot_chain][oxh]
            ox_sc_atoms = [atom for atom in ox_atoms if atom not in ['N','CA','C','O']]

            metrics2measure['rmsd'][f'ox{k+1}_sc_rmsd_toref'] = \
                    {'name': f'ox{k+1}_sc_rmsd_toref',
                     'atm_maps1': [{'chain': prot_chain, 'resnum': oxh, 'atom_name': atom}
                                    for atom in ox_sc_atoms],
                     'atm_maps2': [{'chain': prot_chain, 'resnum': oxh, 'atom_name': atom, 'ref': True}
                                    for atom in ox_sc_atoms]}        

        # oxyanion hole uncertainty
        metrics2measure['uncerts'][f'oxh{k+1}DON_unc'] = \
            {'name': f'oxh{k+1}DON_unc', 
             'atm_map': {'chain': prot_chain, 'resnum': oxh, 'atom_name': oxhatom}}

    if step in ['deacyl', 'tet1', 'tet2']:

        metrics2measure['uncerts']['acylMe_unc'] =  \
            {'name': 'acylMe_unc', 
             'atm_map': {'chain': prot_chain, 'resnum': ser_idx, 'atom_name': acylMe_atom}}

        # --------------------------------------------------------------------------------------------------
        # Below - Measures if the acyl group / tetrahedral intermediate is positioned correctly
        # reference structure is the input structure

        # This metric takes some time to set up because you need a 1 to 1 list of atoms from
        # reference structure paired with tetrahedral intermediate

        if step == 'deacyl':
            ref_atoms = ['O1','C1','C10']
            pred_atoms = [oxyanion_atom, acylattack_atom, acylMe_atom]
        else:
            ref_atoms  = RMSD_TO_REF[ligand_name][step][0]
            pred_atoms = RMSD_TO_REF[ligand_name][step][1]

        metrics2measure['rmsd'][f'{step}_geom_rmsd_toreflig'] = \
            {'name': f'{step}_geom_rmsd_toreflig',
             'atm_maps1': [{'chain': prot_chain, 'resnum': ser_idx, 'atom_name': atom} 
                           for atom in pred_atoms],
             'atm_maps2': [{'chain':  lig_chain, 'resnum': lig_idx, 'atom_name': atom, 'ref': True}
                           for atom in ref_atoms]}

        #----------------------------------------------------------------------------------------------------
        # His-LG hbond metrics for tetrahedral intermediates
    
        if step != 'deacyl':
            lg = LG_NAMES[ligand_name][step]
            metrics2measure['distance']['hislg_hbdist'] = \
                {'name': 'hislg_hbdist',
                 'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc, 'ref': False},
                              {'chain': prot_chain, 'resnum': ser_idx, 'atom_name':      lg,  'ref': False}],
                 'compare2ref': False}

            # His - LG angles 
            metrics2measure['angle']['hislg_angleB'] = \
                {'name': 'hislg_angleB',
                 'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc, 'ref': False},
                              {'chain': prot_chain, 'resnum': ser_idx, 'atom_name':      lg,  'ref': False},
                              {'chain': prot_chain, 'resnum': ser_idx, 'atom_name':   'C1A', 'ref': False}],
                 'compare2ref': False}

            metrics2measure['angle']['hislg_angleA'] = \
                {'name': 'hislg_angleA',
                 'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name':   'CE1', 'ref': False},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc, 'ref': False},
                              {'chain': prot_chain, 'resnum': ser_idx, 'atom_name':      lg, 'ref': False}],
                 'compare2ref': False}

            # His - LG hbond dihedral
            metrics2measure['dihedral']['hislg_dihedralA'] = \
                {'name': 'hislg_dihedralA',
                 'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_don, 'ref': False},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name':   'CE1', 'ref': False},
                              {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc, 'ref': False},
                              {'chain': prot_chain, 'resnum': ser_idx, 'atom_name':    lg, 'ref': False}],
                 'compare2ref': True}

        #----------------------------------------------------------------------------------------------------
        # Oxyanion hole metrics

        for k, oxh in enumerate(oxhres):
            resn = oxhresns[k]
            oxhatom = ','.join([atmset[0] for atmset in OXH_ATOMS[resn]])

            # oxyanion hole hbonding distance
            metrics2measure['distance'][f'ox{k+1}_hbdist'] = \
                {'name': f'oxh{k+1}_hbdist',
                 'atm_maps': [{'chain': prot_chain, 'resnum': ser_idx, 'atom_name': oxyanion_atom, 'ref': False},
                              {'chain': prot_chain, 'resnum':     oxh, 'atom_name':       oxhatom, 'ref': False}],
                 'compare2ref': False}

            # oxyanion hole hbonding distance to reference ligand
            metrics2measure['distance'][f'ox{k+1}_hbdist_toreflig'] = \
                {'name': f'oxh{k+1}_hbdist_toreflig',
                 'atm_maps': [{'chain': lig_chain,  'resnum': lig_idx, 'atom_name':    'O1', 'ref': True},
                              {'chain': prot_chain, 'resnum':     oxh, 'atom_name': oxhatom, 'ref': False}],
                 'compare2ref': False}

        #---------------------------------------------------------------------------------------------------

        # uncertainty in oxyanion position
        metrics2measure['uncerts']['oxyan_unc'] = \
            {'name': 'oxyan_unc',  
             'atm_map': {'chain': prot_chain, 'resnum': ser_idx, 'atom_name': oxyanion_atom}}

        # Oxyanion rmsd from substrate in design model
        metrics2measure['rmsd']['oxyanion_rmsd_toreflig'] = \
            {'name': 'oxyanion_rmsd_toreflig',
             'atm_maps1': [{'chain': prot_chain, 'resnum': ser_idx, 'atom_name': oxyanion_atom}],
             'atm_maps2': [{'chain':  lig_chain, 'resnum': lig_idx, 'atom_name': 'O1', 'ref': True}]}

    elif step == 'acyl':

        #----------------------------------------------------------------------------------------------------
        # Oxyanion hole metrics

        for k, oxh in enumerate(oxhres):

            resn = oxhresns[k]
            oxhatom = ','.join([atmset[0] for atmset in OXH_ATOMS[resn]])

            # oxyanion hole hbonding distance
            metrics2measure['distance'][f'oxh{k+1}_hbdist'] = \
                {'name': f'oxh{k+1}_hbdist',
                 'atm_maps': [{'chain': lig_chain,  'resnum': lig_idx, 'atom_name':    'O1', 'ref': False},
                              {'chain': prot_chain, 'resnum':     oxh, 'atom_name': oxhatom, 'ref': False}],
                 'compare2ref': False}

            # oxyanion hole hbonding distance to reference ligand
            metrics2measure['distance'][f'oxh{k+1}_hbdist_toreflig'] = \
                {'name': f'oxh{k+1}_hbdist_toreflig',
                 'atm_maps': [{'chain': lig_chain,  'resnum': lig_idx, 'atom_name':    'O1', 'ref': True},
                              {'chain': prot_chain, 'resnum':     oxh, 'atom_name': oxhatom, 'ref': False}],
                 'compare2ref': False}

        #---------------------------------------------------------------------------------------------------

        # uncertainty in oxyanion position
        metrics2measure['uncerts']['oxyan_unc'] = \
            {'name': 'oxyan_unc',  
             'atm_map': {'chain': lig_chain, 'resnum': lig_idx, 'atom_name': 'O1'}}

        # Oxyanion rmsd from substrate in design model
        metrics2measure['rmsd']['oxyanion_rmsd_toreflig'] = \
            {'name': 'oxyanion_rmsd_toreflig',
             'atm_maps1': [{'chain':  lig_chain, 'resnum': lig_idx, 'atom_name': 'O1'}],
             'atm_maps2': [{'chain':  lig_chain, 'resnum': lig_idx, 'atom_name': 'O1', 'ref': True}]}
    
    elif step == 'free':
        
        for k, oxh in enumerate(oxhres):
            resn = oxhresns[k]
            oxhatom = ','.join([atmset[0] for atmset in OXH_ATOMS[resn]])
        
            # oxyanion hole hbonding distance to reference ligand
            metrics2measure['distance'][f'oxh{k+1}_hbdist_toreflig'] = \
                {'name': f'oxh{k+1}_hbdist_toreflig',
                 'atm_maps': [{'chain': lig_chain,  'resnum': lig_idx, 'atom_name':    'O1', 'ref': True},
                              {'chain': prot_chain, 'resnum':     oxh, 'atom_name': oxhatom, 'ref': False}],
                 'compare2ref': False}


    return metrics2measure

import argparse 
parser = argparse.ArgumentParser()

# required
parser.add_argument("--pdb",    type=str, help="Chemnet model.")
parser.add_argument("--ref",    type=str, help="Reference pdb. Must contain 4MUAc if serine hydrolase.")
parser.add_argument("--csv",    type=str, help="Chemnet csv.")
parser.add_argument("--outcsv", type=str, help="Path to output data.")

# serine hydrolase specific stuff
parser.add_argument("--catres", type=str, help="Catalytic dyad/triad residues in ser-his-asp order. No oxyanion residues.")
parser.add_argument("--oxhres", type=str, help="Oxyanion hole residue(s).")
parser.add_argument("--p_chain",type=str, help="Chain letter of protein part.")
parser.add_argument("--l_chain",type=str, help="Chain letter of ligand part.")
parser.add_argument("--lig_idx",type=int, help="Ligand index (pdb numbering).")
parser.add_argument("--step",   type=str, default='some_step', help="Catalytic step being examined: acyl, tet1, tet2 or deacyl.")
parser.add_argument("--oxh_types", type=str, help="Are oxyanion hole residues thru sidechain or backbone? Specify by comma separated list ex. 'sc,bb'")
parser.add_argument("--num_preds",type=int, default=50, help="Number of chemnet predictions")
#parser.add_argument("--oxh2_is_sc", action='store_true', default=False, help="Use if oxyanion hole #2 is a sidechain. Omit if bb amide.")
parser.add_argument("--acylenzyme_oxyanion", type=str, default='OAC', help="If modeling deacylation: name of oxyanion atom in acylenzyme, default=OAC")
parser.add_argument("--acylenzyme_methyl",   type=str, default='C2A', help="If modeling deacylation: name of the C atom adjacent to the C that was attacked by Ser to form acylenzyme, default=C2A")
parser.add_argument("--acylenzyme_attackC",  type=str, default='C1A', help="If modeling deacylation: name of C atom attacked by Ser to form acylenzyme, default=C1A")
parser.add_argument("--nuc",  type=str, default='ser', help="Nucleophile reside name ('ser' or 'cys' currently supported)")
parser.add_argument("--his_donor_atom",  type=str, default='ND1', help="Atom name of donor atom of histidine general acid/base (the one hbonded to acid).")
parser.add_argument("--his_acceptor_atom",  type=str, default='NE2', help="Atom name of acceptor atom of histidine general acid/base (the one hbonded to serine).")
parser.add_argument("--debug", action="store_true", default=False, help="Print useful stuff")

args = parser.parse_args()
parser.set_defaults()
print("Using the following arguments:")
print(args)

catres = [int(x) for x in args.catres.split(',')]
if len(catres) == 2:
    catres.append(None)

oxhres = [int(x) for x in args.oxhres.split(',')]
if len(oxhres) == 1:
    oxhres.append(None)

# open the first prediction so we can use some info to determine the key atoms, etc
p0 = Prediction(pdb=args.pdb, csv=args.csv, i=1, step=args.step, ref=args.ref)

# figure out identities of oxyanion holes if sidechain
oxhresns = []
oxhtypes = args.oxh_types.split(',')
for oxh, oxhtype in zip(oxhres, oxhtypes):
    ic(oxhtype)
    if oxhtype == 'sc':
        for res in p0.model:
            if res.atom_records[0].resSeq == oxh and res.atom_records[0].chainID == args.p_chain:
                oxhresn = aa_3_1[res.atom_records[0].resName]
    else:
        oxhresn = 'bb'
    oxhresns.append(oxhresn)
if args.debug:
    ic(oxhresns) 

# figure out the name of the ligand
if args.step == 'free':
    ligand_name = None
else:
    for res in p0.ref_model:
        if res.atom_records[0].resSeq == args.lig_idx and res.atom_records[0].chainID == args.l_chain:
            ligand_name = res.atom_records[0].resName
    if args.debug:
        ic(ligand_name)

# chains
prot_chain = args.p_chain
lig_chain = args.l_chain

metrics2measure = make_metric_dict(ser_idx=catres[0], 
                                   his_idx=catres[1], 
                                   asp_idx=catres[2], 
                                   ox1_idx=oxhres[0], 
                                   ox2_idx=oxhres[1], 
                                   lig_idx=args.lig_idx,
                                   prot_chain=prot_chain, 
                                   lig_chain=lig_chain,
                                   step=args.step, 
                                   oxhresns=oxhresns, 
                                   atomic_info=p0.atomic_info,
                                   oxyanion_atom=args.acylenzyme_oxyanion,
                                   acylMe_atom=args.acylenzyme_methyl,
                                   acylattack_atom=args.acylenzyme_attackC,
                                   ligand_name=ligand_name,
                                   nuc=args.nuc,
                                   his_acc=args.his_acceptor_atom,
                                   his_don=args.his_donor_atom
                                   )


# storing measured values
metrics_measured = {}
for metric_type in metrics2measure.keys():
    for k in metrics2measure[metric_type]:
        metric = metrics2measure[metric_type][k]
        metrics_measured[metric['name']] = []
        metrics_measured[metric['name']+'_ref'] = []


####################### KEY HYDROGEN BONDS ###############################################################################

HBONDS = {
    'ser-his':   ['serhis_hbdist', 'serhis_angleA', 'serhis_angleB', 'serhis_dihedralB'],
    'his-lgref': ['hislg_hbdist_toreflig', 'hislg_angleA_toreflig', 'hislg_angleB_toreflig', 'hislg_dihedralA_toreflig'],
}

# oxyanion hole hbonds
oxhres = [res for res in oxhres if res]
for i, oxh in enumerate(oxhres):
    if args.step != 'free':
        HBONDS[f'oxh{i+1}-ox']    = [f'oxh{i+1}_hbdist', f'oxh{i+1}_angleA', f'oxh{i+1}_angleB', f'oxh{i+1}_dihedralB']
    HBONDS[f'oxh{i+1}-oxref'] = [f'oxh{i+1}_hbdist_toreflig', f'oxh{i+1}_angleA_toreflig', f'oxh{i+1}_angleB_toreflig',
                                 f'oxh{i+1}_dihedralB_toreflig']

# if asp is present
if catres[2]:
    his_idx = catres[1]
    asp_idx = catres[2]
    HBONDS['his-asp'] = ['hisacid_hbdist', 'hisacid_angleA', 'hisacid_angleB', 'hisacid_dihedralA', 'hisacid_dihedralB']

# additional hbond if tetrahedral intermediate
if args.step in ['tet1', 'tet2']:
    HBONDS['his-lg'] = ['hislg_hbdist', 'hislg_angleA', 'hislg_angleB', 'hislg_dihedralA']

# add all the hbonds to the metrics list
for hbond in HBONDS:
    metrics_measured[f'is_hbond_{hbond}'] = []

if args.debug:
    ic(metrics_measured)
    ic(HBONDS)

his_don = args.his_donor_atom
his_acc = args.his_acceptor_atom

########################################################################################################################


interacting_atoms = {}

ligand_metrics = ['fape','lddt','rmsd','kabsch','prmsd','plddt','plddt_pde']
for m in ligand_metrics:
    metrics_measured[m] = []

print(f'measuring...')
for i in range(1,args.num_preds+1):

    p1 = Prediction(pdb=args.pdb, csv=args.csv, i=i, step=args.step, ref=args.ref)

    metrics_measured['fape'].append(p1.fape)
    metrics_measured['lddt'].append(p1.lddt)
    metrics_measured['rmsd'] .append(p1.rmsd)
    metrics_measured['kabsch'].append(p1.kabsch)
    metrics_measured['prmsd'].append(p1.prmsd)
    metrics_measured['plddt'].append(p1.plddt)
    metrics_measured['plddt_pde'].append(p1.plddt_pde)
    
    if args.debug:
        print(f'----model {i}----')
    for k, metric in metrics2measure['distance'].items():
        if args.debug:
            print(metric['name'])
        distance, atmpair = p1.get_distance(metric['atm_maps'][0], metric['atm_maps'][1])
        if metric['compare2ref']:
            ref_distance, atmpair = p1.get_distance(metric['atm_maps'][0], metric['atm_maps'][1], ref=True)
            
        else:
            ref_distance = None

        metrics_measured[metric['name']].append(distance)
        metrics_measured[metric['name']+'_ref'].append(ref_distance)
        if args.debug:
            print(distance, ref_distance)

        # for measuring angles later to know which atoms are interacting
        for k,v in HBONDS.items():
            if metric['name'] == v[0]:
                interacting_atoms[k] = atmpair
    
    if args.debug:
        ic(interacting_atoms)

    # add oxyanion hole hbond angles now that we know which atoms are interacting
    new_metrics = []
    # ic(oxhres)
    for k, oxh in enumerate(oxhres):

        # add these metrics to output dictionary
        for m in ['_angleA', '_angleB', '_dihedralB']:
            new_metrics.append(f'oxh{k+1}{m}')
            new_metrics.append(f'oxh{k+1}{m}_toreflig')
        if i == 1:
            for m in new_metrics:
                metrics_measured[m] = []
                metrics_measured[m+'_ref'] = []

        # get interacting atom from oxyanion hole
        if args.step == 'free':
            ligatm, oxhatm1 = interacting_atoms[f'oxh{k+1}-oxref']
        else:
            ligatm, oxhatm1 = interacting_atoms[f'oxh{k+1}-ox']
        
        if args.debug:
            ic(ligatm, oxhatm1)

        # find dihedral for that atom
        resn = oxhresns[k]
        for datms in OXH_ATOMS[resn]:
            if datms[0] == oxhatm1:
                oxh_dihedral_atms = datms

        oxhatm2 = oxh_dihedral_atms[1]
        oxhatm3 = oxh_dihedral_atms[2]
        
        if args.step != 'free':
            # depending on the step, ligand atoms are different.
            if args.step == 'acyl':
                ligatm1, ligatm2, ligatm3 = 'O1', 'C1', 'O2'
                oxlig_chain = lig_chain
                oxlig_idx = args.lig_idx
            else:
                ligatm1, ligatm2, ligatm3 = args.acylenzyme_oxyanion, args.acylenzyme_attackC, args.acylenzyme_methyl
                oxlig_chain = prot_chain
                oxlig_idx = catres[0] # serine
            
            metrics2measure['angle'][f'oxh{k+1}_angleA'] = \
                {'name': f'oxh{k+1}_angleA',
                 'atm_maps': [{'chain': oxlig_chain, 'resnum': oxlig_idx, 'atom_name': ligatm2, 'ref': False},
                              {'chain': oxlig_chain, 'resnum': oxlig_idx, 'atom_name': ligatm1, 'ref': False},
                              {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm1, 'ref': False}],
                 'compare2ref': False}
            metrics2measure['angle'][f'oxh{k+1}_angleB'] = \
                {'name': f'oxh{k+1}_angleB',
                 'atm_maps': [{'chain': oxlig_chain, 'resnum': oxlig_idx, 'atom_name': ligatm1, 'ref': False},
                              {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm1, 'ref': False},
                              {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm2, 'ref': False}],
                 'compare2ref': False}
            #metrics2measure['dihedral'][f'oxh{k+1}_dihedralA'] = \
            #    {'name': f'oxh{k+1}_dihedralA',
            #     'atm_maps': [{'chain': oxlig_chain, 'resnum': oxlig_idx, 'atom_name': ligatm3, 'ref': False},
            #                  {'chain': oxlig_chain, 'resnum': oxlig_idx, 'atom_name': ligatm2, 'ref': False},
            #                  {'chain': oxlig_chain, 'resnum': oxlig_idx, 'atom_name': ligatm1, 'ref': False},
            #                  {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm1, 'ref': False}],
            #     'compare2ref': False}
            metrics2measure['dihedral'][f'oxh{k+1}_dihedralB'] = \
                {'name': f'oxh{k+1}_dihedralB',
                 'atm_maps': [{'chain': oxlig_chain, 'resnum': oxlig_idx, 'atom_name': ligatm1, 'ref': False},
                              {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm1, 'ref': False},
                              {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm2, 'ref': False},
                              {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm3, 'ref': False}],
                 'compare2ref': False}
       
        # reference angles
        lig_idx = args.lig_idx
        metrics2measure['angle'][f'oxh{k+1}_angleA_toreflig'] = \
            {'name': f'oxh{k+1}_angleA_toreflig',
             'atm_maps': [{'chain':   lig_chain, 'resnum':   lig_idx, 'atom_name':    'C1', 'ref':  True},
                          {'chain':   lig_chain, 'resnum':   lig_idx, 'atom_name':    'O1', 'ref':  True},
                          {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm1, 'ref': False}],
             'compare2ref': False}
        metrics2measure['angle'][f'oxh{k+1}_angleB_toreflig'] = \
            {'name': f'oxh{k+1}_angleB_toreflig',
             'atm_maps': [{'chain':   lig_chain, 'resnum':   lig_idx, 'atom_name':    'O1', 'ref':  True},
                          {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm1, 'ref': False},
                          {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm2, 'ref': False}],
             'compare2ref': False}
        #metrics2measure['dihedral'][f'oxh{k+1}_dihedralA_toreflig'] = \
        #    {'name': f'oxh{k+1}_dihedralA_toreflig',
        #     'atm_maps': [{'chain':   lig_chain, 'resnum':   lig_idx, 'atom_name':    'O2', 'ref':  True},
        #                  {'chain':   lig_chain, 'resnum':   lig_idx, 'atom_name':    'C1', 'ref':  True},
        #                  {'chain':   lig_chain, 'resnum':   lig_idx, 'atom_name':    'O1', 'ref':  True},
        #                  {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm1, 'ref': False}],
        #     'compare2ref': False}
        metrics2measure['dihedral'][f'oxh{k+1}_dihedralB_toreflig'] = \
            {'name': f'oxh{k+1}_dihedralB_toreflig',
             'atm_maps': [{'chain':   lig_chain, 'resnum':   lig_idx, 'atom_name':    'O1', 'ref':  True},
                          {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm1, 'ref': False},
                          {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm2, 'ref': False},
                          {'chain':  prot_chain, 'resnum':       oxh, 'atom_name': oxhatm3, 'ref': False}],
             'compare2ref': False}

    # if triad, add asp hbond angles now that we know which atoms are interacting
    if catres[2]:

        # get the interacting atoms
        hisatm, aspatm = interacting_atoms['his-asp']
        if aspatm in ['OD1', 'OD2']:
            chain = ['CG','CB']
        elif aspatm in ['OE1', 'OE2']:
            chain = ['CD','CG']

        # add new metrics to output dictionary
        new_metrics = ['hisacid_angleA', 'hisacid_angleB', 'hisacid_dihedralA', 'hisacid_dihedralB']
        if i == 1:
            for m in new_metrics:
                metrics_measured[m] = []
                metrics_measured[m+'_ref'] = []

        metrics2measure['angle']['hisacid_angleA'] = \
            {'name': 'hisacid_angleA',
             'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name':  'CE1', 'ref': False},
                          {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_don, 'ref': False},
                          {'chain': prot_chain, 'resnum': asp_idx, 'atom_name': aspatm, 'ref': False}],
             'compare2ref': False}
        
        metrics2measure['angle']['hisacid_angleB'] = \
            {'name': 'hisacid_angleB',
             'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name':  his_don, 'ref': False},
                          {'chain': prot_chain, 'resnum': asp_idx, 'atom_name':   aspatm, 'ref': False},
                          {'chain': prot_chain, 'resnum': asp_idx, 'atom_name': chain[0], 'ref': False}],
             'compare2ref': False}

        metrics2measure['dihedral']['hisacid_dihedralA'] = \
            {'name': 'hisacid_dihedralA',
             'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_acc, 'ref': False},
                          {'chain': prot_chain, 'resnum': his_idx, 'atom_name':   'CE1', 'ref': False},
                          {'chain': prot_chain, 'resnum': his_idx, 'atom_name': his_don, 'ref': False},
                          {'chain': prot_chain, 'resnum': asp_idx, 'atom_name':  aspatm, 'ref': False}],
             'compare2ref': False}
        
        metrics2measure['dihedral']['hisacid_dihedralB'] = \
            {'name': 'hisacid_dihedralB',
             'atm_maps': [{'chain': prot_chain, 'resnum': his_idx, 'atom_name':  his_don, 'ref': False},
                          {'chain': prot_chain, 'resnum': asp_idx, 'atom_name':   aspatm, 'ref': False},
                          {'chain': prot_chain, 'resnum': asp_idx, 'atom_name': chain[0], 'ref': False},
                          {'chain': prot_chain, 'resnum': asp_idx, 'atom_name': chain[1], 'ref': False}],
             'compare2ref': False}
    
    # Measure angles
    for k, metric in metrics2measure['angle'].items():
        if args.debug:
            print(metric['name'])
        angle = p1.get_angle(metric['atm_maps'][0], metric['atm_maps'][1], metric['atm_maps'][2])
        if metric['compare2ref']:
            ref_angle = p1.get_angle(metric['atm_maps'][0], metric['atm_maps'][1], metric['atm_maps'][2], ref=True)
        else:
            ref_angle = None
        
        metrics_measured[metric['name']].append(angle)
        metrics_measured[metric['name']+'_ref'].append(ref_angle)
        if args.debug:
            print(angle, ref_angle)

    # Measure dihedrals
    for k, metric in metrics2measure['dihedral'].items():
        if args.debug:
            print(metric['name'])
        angle = p1.get_dihedral(metric['atm_maps'][0], metric['atm_maps'][1], metric['atm_maps'][2], metric['atm_maps'][3])
        if metric['compare2ref']:
            ref_angle = p1.get_dihedral(metric['atm_maps'][0], metric['atm_maps'][1], metric['atm_maps'][2], metric['atm_maps'][3], ref=True)
        else:
            ref_angle = None

        metrics_measured[metric['name']].append(angle)
        metrics_measured[metric['name']+'_ref'].append(ref_angle)
        if args.debug:
            print(angle, ref_angle)
    
    # Collect uncertainties
    for k, metric in metrics2measure['uncerts'].items():
        if args.debug:
            print(metric['name'])
        unc = p1.get_atom_uncertainty(metric['atm_map'])
        metrics_measured[metric['name']].append(unc)
        if args.debug:
            print(unc)

    # Measure RMSDs
    for k, metric in metrics2measure['rmsd'].items():
        if args.debug:
            print(metric['name'])
        rmsd = p1.get_rmsd(metric['atm_maps1'], metric['atm_maps2'])
        metrics_measured[metric['name']].append(rmsd)
        if args.debug:
            print(rmsd)


    # Assess hydrogen bonds
    for hbond in HBONDS:
        

        # get hybridization for each atom
        atmpair = interacting_atoms[hbond]
        atmhybr = [k for k,v in ATM_HYBRID.items() for atm in atmpair if atm in v]
        assert len(atmpair) == len(atmhybr)
        
        if args.debug:
            ic(f'ASSESSING HBOND {hbond}')
            ic(atmpair)
            ic(atmhybr)
    
        # unpack values
        distance_name = None
        angleA_name = None
        angleB_name = None 
        dihedralA_name = None
        dihedralAB_name = None
        dihedralB_name = None
    
        for hbond_param in HBONDS[hbond]:
            if 'hbdist' in hbond_param:
                distance_name = hbond_param
            elif 'angleA' in hbond_param:
                angleA_name = hbond_param
            elif 'angleB' in hbond_param:
                angleB_name = hbond_param
            elif 'dihedralA' in hbond_param:
                dihedralA_name = hbond_param
            elif 'dihedralAB' in hbond_param:
                dihedralAB_name = hbond_param
            elif 'dihedralB' in hbond_param:
                dihedralB_name = hbond_param
            else:
                raise Exception
    
        distance   = metrics_measured[distance_name][-1] if distance_name else None
        angleA     = metrics_measured[angleA_name][-1] if angleA_name else None
        angleB     = metrics_measured[angleB_name][-1] if angleB_name else None
        dihedralA  = metrics_measured[dihedralA_name][-1] if dihedralA_name else None
        dihedralAB = metrics_measured[dihedralAB_name][-1] if dihedralAB_name else None
        dihedralB  = metrics_measured[dihedralB_name][-1] if dihedralB_name else None

        # look up cutoffs based on atom hybridization
        # for angleB, dihedralB, need to use atom2 hybridization to pick cutoff
        # atom1 for angleA and dihedralA
        distance_cuts = HB_CUTOFFS[atmhybr[0]]['distance']
        angleA_cuts   = HB_CUTOFFS[atmhybr[0]]['angleA']
        angleB_cuts   = HB_CUTOFFS[atmhybr[1]]['angleA']

        # if sidechain is planar, also add the dihedral filters
        # arg, his, trp, asp, glu, asn, gln 
        # first take absolute value of dihedral angle so cutoffs look different
        if args.debug:
            ic(atmpair[0], atmpair[1])


        dihedralA_cuts = None
        dihedralB_cuts = None

        if atmpair[0] in ATMS_PLANAR_0_180:
            dihedralA_cuts = (130.0, 50.0)
        elif atmpair[0] in ATMS_PLANAR_180:
            dihedralA_cuts = (130.0, None)
        if atmpair[1] in ATMS_PLANAR_0_180:
            dihedralB_cuts = (130.0, 50.0)
        elif atmpair[1] in ATMS_PLANAR_180:
            dihedralB_cuts = (130.0, None)


        # check if geometry matches
        is_hbond = True
        if not (distance >= distance_cuts[0] and distance <= distance_cuts[1]):
            if args.debug:
                ic('hbond failed distance filter')
            is_hbond = False
        if not (angleA >= angleA_cuts[0] and angleA <= angleA_cuts[1]):
            if args.debug:
                ic('hbond failed angleA filter')
            is_hbond = False
        if not (angleB >= angleB_cuts[0] and angleB <= angleB_cuts[1]):
            if args.debug:
                ic('hbond failed angleB filter')
            is_hbond = False

        # dihedral angles are complicated...
        # have to take the absolute value
        # dihedral could be 0 or 180 depending on atom
        if dihedralA_cuts:
            dihedralA_abs = abs(dihedralA)
            if not dihedralA_abs >= dihedralA_cuts[0]:
                # if either this is a 180 degree cutoff only or it doesn't pass the 0 degree cutoff
                # mark as false
                if not dihedralA_cuts[1] or not dihedralA_abs <= dihedralA_cuts[1]:
                    if args.debug:
                        ic('hbond failed dihedralA 0 and dihedralA 1 filter')
                    is_hbond = False

        if dihedralB_cuts:
            dihedralB_abs = abs(dihedralB)
            if not dihedralB_abs >= dihedralB_cuts[0]:
                # if either this is a 180 degree cutoff only or it doesn't pass the 0 degree cutoff
                # mark as false
                if not dihedralB_cuts[1] or not dihedralB_abs <= dihedralB_cuts[1]:
                    if args.debug:
                        ic('hbond failed dihedralB 0 and dihedralB 1 filter')
                    is_hbond = False

        
        metrics_measured[f'is_hbond_{hbond}'].append(is_hbond)
        
        if args.debug:

            ic(distance)
            ic(angleA)
            ic(angleB)
            ic(dihedralA)
            ic(dihedralAB)
            ic(dihedralB)

            ic(distance_cuts)
            ic(angleA_cuts)
            ic(angleB_cuts)
            ic(dihedralA_cuts)
            ic(dihedralB_cuts)

            ic(is_hbond) 

for key in list(metrics_measured.keys()):
    if len(metrics_measured[key]) == 0:
        metrics_measured.pop(key)
        continue
    elif not all(metrics_measured[key]) and 'is_hbond' not in key:
        metrics_measured.pop(key)
        continue

metrics_measured['pdb']    = [p1.pdb]*50
metrics_measured['ref']    = [p1.ref]*50
metrics_measured['step']   = [p1.step]*50
metrics_measured['csv']    = [p1.csv]*50
metrics_measured['i']      = list(range(50))
metrics_measured['label']  = [p1.label]*50

# debugging
if args.debug:
    for k in metrics_measured:
        print(k, len(metrics_measured[k]))
        if len(metrics_measured[k]) != 50:
            print(k, 'is bad!')

df = pd.DataFrame.from_dict(metrics_measured)
df.to_csv(args.outcsv, index=False)

print('finished!')

