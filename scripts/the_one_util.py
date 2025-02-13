import pyrosetta
import numpy as np

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

def mutate(pose, resi, resn):
    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
    mutres.set_target(resi)
    mutres.set_res_name(resn)
    mutres.apply(pose) 
    return pose
