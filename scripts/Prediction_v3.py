
import sys, glob, os
from SimplePdbLib import *
import geometry
import pandas as pd
import numpy as np
from icecream import ic

class Prediction:

    def __init__(self, pdb, i, step, csv, ref):

        self.pdb = pdb
        self.ref = ref # the pdb chemnet was run on.

        self.csv = csv
        self.step = step

        # open pdb and get the i'th prediction, 0-index
        self.i = i
        self.model = read_in_stubs_file(self.pdb)[self.i-1]
        self.ref_model = read_in_stubs_file(self.ref)[0]

        # parse through the SimplePdbLib object for atomic coordinates & uncertainties
        self.atomic_info = self.collect_atomic_info(self.model)
        self.ref_atomic_info = self.collect_atomic_info(self.ref_model)

        # save the ligand metrics also
        ligand_df = pd.read_csv(csv)
        metrics = ligand_df[ligand_df['model_idx'] == self.i]

        self.label  = metrics['label'].squeeze()
        self.fape   = metrics['fape'].squeeze()
        self.lddt   = metrics['lddt'].squeeze()
        self.rmsd   = metrics['rmsd'].squeeze()
        self.kabsch = metrics['kabsch'].squeeze()
        self.prmsd  = metrics['prmsd'].squeeze()
        self.plddt  = metrics['plddt'].squeeze()
        self.plddt_pde  = metrics['plddt_pde'].squeeze()


    def collect_atomic_info(self, model):
        """
        Acts on SimplePdb Object, creates dict of {chain: {resi: {atom_name: (xyz, uncertainty)}}
        """
        atomic_info = {}

        for res in model:
            for atom, coords in zip(res.atom_records, res.coords):

                if atom.chainID not in atomic_info.keys():
                    atomic_info[atom.chainID] = {}
                if atom.resSeq not in atomic_info[atom.chainID].keys():
                    atomic_info[atom.chainID][atom.resSeq] = {}

                atomic_info[atom.chainID][atom.resSeq][atom.name] = (coords, atom.tempFactor)

        return atomic_info

    def get_atom_coordinates(self, atm_map, ref=False):
        """
        Atom map format:
        {'chain': chain, 'resnum', resnum, 'atom_name': atom_name}
        if ref=True, get coordinates from reference model, not prediction.
        """
        chain = atm_map['chain']
        resnum = atm_map['resnum']
        name = atm_map['atom_name']

        if 'ref' in atm_map.keys():
            ref = atm_map['ref']
        if ref:
            xyz_coords = self.ref_atomic_info[chain][resnum][name][0]
        else:
            xyz_coords = self.atomic_info[chain][resnum][name][0]

        return xyz_coords

    def get_atom_uncertainty(self, atm_map):
        # ic(atm_map)
        chain = atm_map['chain']
        resnum = atm_map['resnum']

        uncs = []
        for name in atm_map['atom_name'].split(','):
            unc = self.atomic_info[chain][resnum][name][1]
            uncs.append(unc)
        mean = sum(uncs) / len(uncs)
        return round(mean, 2)

    def get_distance(self, atm1_map, atm2_map, ref=False):
        # map structure: {'chain', resnum, 'atom_name'}

        # cycle through all the atoms in the atom maps
        # majority of the time there will only be one
        distances = []
        for atm1_atom in atm1_map['atom_name'].split(','):
            
            # rebuild the atom map for this atom
            atm1_map_atom = {'chain': atm1_map['chain'],
                             'resnum': atm1_map['resnum'],
                             'atom_name': atm1_atom,
                             'ref': atm1_map['ref']}

            for atm2_atom in atm2_map['atom_name'].split(','):

                # rebuild the atom map for this atom
                atm2_map_atom = {'chain': atm2_map['chain'],
                                 'resnum': atm2_map['resnum'],
                                 'atom_name': atm2_atom,
                                 'ref': atm2_map['ref']}

                atm1_xyz = self.get_atom_coordinates(atm1_map_atom, ref=atm1_map_atom['ref'])
                atm2_xyz = self.get_atom_coordinates(atm2_map_atom, ref=atm2_map_atom['ref'])

                if ref:
                    atm1_map_atom['ref'] = True
                    atm2_map_atom['ref'] = True
                    atm1_xyz = self.get_atom_coordinates(atm1_map_atom, ref=ref)
                    atm2_xyz = self.get_atom_coordinates(atm2_map_atom, ref=ref)


                distance = geometry.measure_distance(atm1_xyz, atm2_xyz)
                distances.append(distance)
    
        # determine the minimum distance and the corresponding atoms
        min_distance = min(distances)
        min_distance_idx = distances.index(min_distance)
        
        # magical chatgpt stuff
        num_atm2s = len(atm2_map['atom_name'].split(','))
        min_atm1_idx = min_distance_idx // num_atm2s
        min_atm2_idx = min_distance_idx % num_atm2s

        min_atm1 = atm1_map['atom_name'].split(',')[min_atm1_idx]
        min_atm2 = atm2_map['atom_name'].split(',')[min_atm2_idx]
        
        # returns the minimum distance and the atom pair in that distance
        return round(min_distance, 3), (min_atm1, min_atm2)

    def get_angle(self, atm1_map, atm2_map, atm3_map, ref=False):
        # in v3:
        # this function measures distance and then uses the interacting atoms to
        # calculate angles

        atm1_xyz = self.get_atom_coordinates(atm1_map, ref=ref)
        atm2_xyz = self.get_atom_coordinates(atm2_map, ref=ref)
        atm3_xyz = self.get_atom_coordinates(atm3_map, ref=ref)

        angle = geometry.measure_angle(atm1_xyz, atm2_xyz, atm3_xyz)
        return round(angle, 3)

    def get_dihedral(self, atm1_map, atm2_map, atm3_map, atm4_map, ref=False):

        atm1_xyz = self.get_atom_coordinates(atm1_map, ref=ref)
        atm2_xyz = self.get_atom_coordinates(atm2_map, ref=ref)
        atm3_xyz = self.get_atom_coordinates(atm3_map, ref=ref)
        atm4_xyz = self.get_atom_coordinates(atm4_map, ref=ref)

        dihedral = geometry.measure_dihedral(atm1_xyz, atm2_xyz, atm3_xyz, atm4_xyz)
        return round(dihedral, 3)

    def get_rmsd(self, atm_maps1, atm_maps2):
        coords1 = np.array([self.get_atom_coordinates(atm_maps1[j]) 
                            for j in range(len(atm_maps1))])
        coords2 = np.array([self.get_atom_coordinates(atm_maps2[j]) 
                            for j in range(len(atm_maps2))])
        rmsd = RMSD(coords1, coords2)
        return round(rmsd, 3)
