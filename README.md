# serine-hydrolase-design

Code repository for de novo design of serine hydrolases, to accompany the manuscript:

Lauko A, Pellock SJ. 2024. Computational design of serine hydrolases. *bioRxiv* doi: 10.1101/2024.08.29.610411


## A. Motif Generation
RFdiffusion takes as input motifs consisting of "stubs" that hold the catalytic residues. To avoid constraining design to existing serine hydrolase active sites, we generate our own.

### Step 01: Sampling histidine stubs.

The script "invrotzyme.py" samples the interactions between the substrate, serine motif, and histidine motif according to the provided constraint file, producing an array of serine hydrolase active sites. In this case, high probability histidine rotamers on 3-residue helical stubs are being sampled in hydrogen bonding geometries to the serine.

Inputs:
(a) mu1.params - Rosetta params file describing the substrate in a near-tetrahedral geometry.
(b) 1LNS.pdb - PDB file containing the local structure around serine nucleophile from a natural hydrolase (1lns).
(c) 1LNS_mu1.cst - Rosetta enzyme constraint file describing the geometry of the serine-substrate and the serine-histidine interactions.

Outputs:
PDB files containing serine hydrolase active sites.

### Step 02: Increasing motif complexity by diffusion

To add additional residues to the motifs (ex. Asp/Glu triad residue, sidechain oxyanion hole residue), we use RFdiffusion with simple Ser-His motifs generated in step 01 to create new backbones. We then refine the outputs, and proceed to search the generated backbones for positions that can accomodate the aforementioned catalytic motifs.

### Step 02a:
RFdiffusion to generate new backbones around a simple motif.

Inputs:
(a) simple_motif.pdb - PDB file produced in Step 01 containing substrate, catalytic serine, and histidine.

Outputs:
PDB file of diffusion backbone.

### Step 02b: Refinement

Inputs:
(a) simple_motif_diffusion_output.pdb - PDB file of CA trace produced by RFdiffusion in step02a.

Outputs:
PDB file of refined diffusion output.

### Step 02c: Search for positions that accomodate new catalytic elements

In this case we search a backbone for positions in which an acceptor Asp/Glu can make an H-bond with atom ND1 of the catalytic histidine.

Inputs:
(a) Refined diffusion output.
TODO: finish this

## B. Design Pipeline

With motif(s) in hand, we move on to design, which consists of RFdiffusion starting from the motif, refinement of the diffusion outputs, sequence design with LigandMPNN+FastRelax, structural validation with AlphaFold2, multi-step Chemnet prediction, and analysis of Chemnet outputs.

### Step 01: CA RFdiffusion

New backbones are generated that contain the input active site.

Inputs:
(a) PDB file of input motif
(b) Contig input (TODO)

Outputs:
(a) PDB file of diffused CA trace and motif
(b) TRB file containing mapping of input motif sequence positions to diffusion output sequence positions

### Step 02: Refinement

The backbones generated in step 1 are only CA traces. Refinement generates all-atom models from the CA trace which can then be input to LigandMPNN and Rosetta.

Inputs:
(a) PDB file of diffused CA trace
(b) TODO: ???

Outputs:
PDB file of refined all-atom RFdiffusion output

### Step 03: Design with LigandMPNN+FastRelax

The sequences of RFdiffusion outputs are designed with LigandMPNN, then relaxed using Rosetta with enzyme constraints to hold the active site in place. The process is repeated three times.

Inputs:
(a) PDB file of refinement output, or an existing design model if performing sequence resampling.
(b) Rosetta enzyme constraint file specifying desired interaction geometries in active site.
(c) Rosetta params file of substrate.
(d) Locations of catalytic residues in sequence.
TODO

Outputs:
(a) PDB file of design output.
(b) CSV file containing design scores.

### Step 04: Structural validation with AlphaFold2

See AF2 repo TODO



