
import pandas as pd
import pyrosetta

def get_hbonds(pose, exclude_bb=False, exclude_scb=False, exclude_bsc=False, exclude_sc=False):
    """
    returns dataframe of all hbonds in the pose

    exclude_bb (bool): Exclude backbone–backbone hydrogen bonds from the returned HBondSet.
    Defaults to False.

    exclude_bsc (bool): Exclude backbone–side-chain hydrogen bonds from the returned HBondSet.
    Defaults to False.

    exclude_scb (bool): Exclude side-chain–backbone hydrogen bonds from the returned HBondSet.
    Defaults to False.

    exclude_sc (bool): Exclude side-chain hydrogen bonds from the returned HBondSet.
    Defaults to False.

    """


    # Get all hydrogen bonds in the pose
    hbonds = pose.get_hbonds(exclude_bb=exclude_bb, exclude_scb=exclude_scb,
                             exclude_bsc=exclude_bsc, exclude_sc=exclude_sc)
    # Create an empty DataFrame to store the hydrogen bond information
    df = pd.DataFrame(columns=['donor_residue', 'donor_atom', 'acceptor_residue', 'acceptor_atom', 'distance'])

    # create empty lists to store the data
    donor_residues = []
    acceptor_residues = []
    donor_atoms = []
    acceptor_atoms = []
    hbond_energies = []

    # loop over each HBond in the HBondSet and extract the information
    for hbond in hbonds.hbonds():
        donor_residues.append(hbond.don_res())
        acceptor_residues.append(hbond.acc_res())
        donor_atoms.append(pose.residue(hbond.don_res()).atom_name(hbond.don_hatm()))
        acceptor_atoms.append(pose.residue(hbond.acc_res()).atom_name(hbond.acc_atm()))
        hbond_energies.append(hbond.energy())


    # create the pandas dataframe from the lists
    hbond_df = pd.DataFrame({
        'donor_residue': donor_residues,
        'acceptor_residue': acceptor_residues,
        'donor_atom': donor_atoms,
        'acceptor_atom': acceptor_atoms,
        'energy': hbond_energies
    })

    # remove string whitespace in relevant columns
    string_columns = ['donor_atom', 'acceptor_atom']
    if len(hbond_df) != 0:
        for column in string_columns:
            hbond_df[column] = hbond_df[column].str.strip()
    
    return hbond_df

def get_hbonds_to_atom(hbond_df, atom_name, residue_num):
    hbonds_to_res = get_hbonds_to_res(hbond_df, residue_num)
    hbonds_to_atom = hbonds_to_res[(hbonds_to_res['acceptor_atom'] == atom_name) | (hbonds_to_res['donor_atom'] == atom_name)]
    return hbonds_to_atom

def get_hbonds_to_res(hbond_df, residue_num):
    hbonds_to_res = hbond_df[(hbond_df['acceptor_residue'] == residue_num) | (hbond_df['donor_residue'] == residue_num)]
    return hbonds_to_res

def get_resis_hbonded_to_atom(pose, atom_name, resnum):
    """
    Detects all the hbonds in a pose, selects the ones interacting
    with the specified atom in residue of interest, returns a list
    of donors and list of acceptor residue indices (1-indexed).
    """

    # get all hbonds
    hbond_df = get_hbonds(pose, exclude_bb=False)

    # get acceptors and donors to atom
    hbonds_to_atom = get_hbonds_to_atom(hbond_df, atom_name=atom_name, residue_num=resnum)
    # print(hbonds_to_atom)
    # hbonds_to_res = get_hbonds_to_res(hbond_df, residue_num=resnum)
    # print(hbonds_to_res)
    acceptors = [res for res in hbonds_to_atom.acceptor_residue.to_list() if res != resnum]
    donors = [res for res in hbonds_to_atom.donor_residue.to_list() if res != resnum]
    
    return donors, acceptors

def get_atoms_hbonded_to_atom(pose, atom_name, resnum):
    """
    Detects all the hbonds in a pose, selects the ones interacting
    with the specified atom in residue of interest, returns a list
    of donor atoms and acceptor atoms.
    """

    # get all hbonds
    hbond_df = get_hbonds(pose, exclude_bb=False)

    # get acceptors and donors to atom
    hbonds_to_atom = get_hbonds_to_atom(hbond_df, atom_name=atom_name, residue_num=resnum)
    # print(hbonds_to_atom)
    # hbonds_to_res = get_hbonds_to_res(hbond_df, residue_num=resnum)
    # print(hbonds_to_res)
    acceptors = hbonds_to_atom[hbonds_to_atom.acceptor_residue == resnum]
    donors    = hbonds_to_atom[hbonds_to_atom.donor_residue == resnum]

    acceptor_atoms = hbonds_to_atom.acceptor_atom.to_list()
    donor_atoms    = hbonds_to_atom.donor_atom.to_list()
    
    return donor_atoms, acceptor_atoms

def get_resis_hbonded_to_sc(pose, resnum, sc_only=False):
    """
    Detects all the hbonds in a pose, excluding backbone-backbone
    hbonds. Selects the ones interacting with the residue of interest, 
    returns a list of acceptors and list of donor residue indices (1-indexed).

    set sc_only to true to detect only sidechain-sidechain hbonds
    """

    # get all hbonds to and from sidechains
    if sc_only:
        hbond_df = get_hbonds(pose, exclude_bb=True, exclude_scb=True, exclude_bsc=True)
    else:
        hbond_df = get_hbonds(pose, exclude_bb=True)
    
    # get acceptors and donors to sc
    hbonds_to_res = get_hbonds_to_res(hbond_df, residue_num=resnum)
    acceptors = [res for res in hbonds_to_res.acceptor_residue.to_list() if res != resnum]
    donors = [res for res in hbonds_to_res.donor_residue.to_list() if res != resnum]
    
    return acceptors, donors

def get_resis_hbonded_to_sc_thru_bbamide(pose, resnum):
    """
    A function that gets the backbone amide hydrogen bond donors to the
    sidechain of a residue of interest.
    """
        
    hbond_df = get_hbonds(pose, exclude_bb=True, exclude_sc=True)
    hbonds_to_res = get_hbonds_to_res(hbond_df, residue_num=resnum)

    # get only the donors, which should only come from bb H's
    don_to_res = hbonds_to_res[hbonds_to_res['acceptor_residue'] == resnum]
    hbonds_thru_N = hbonds_to_res[hbonds_to_res['donor_atom'] == 'H']
    donors = [res for res in hbonds_thru_N.donor_residue.to_list() if res != resnum]

    return donors 


