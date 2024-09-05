# serine-hydrolase-design

Code repository for de novo design of serine hydrolases, to accompany the manuscript:

Lauko A, Pellock SJ. 2024. Computational design of serine hydrolases. *bioRxiv* doi: 10.1101/2024.08.29.610411

Installing Apptainer (https://apptainer.org/docs/admin/main/installation.html) will allow you to run all code using the pre-packaged python environments found in containers/.


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
(b) TRB file containing mapping of input motif sequence positions to diffusion output sequence positions and input arguments (needed for refinement)

### Step 02: Refinement

The backbones generated in step 1 are only CA traces. Refinement generates all-atom models from the CA trace which can then be input to LigandMPNN and Rosetta. Refinement uses the RFdiffusion runscript in a different mode, and outputs the refined PDB files into the same directory as the CA traces.

Inputs:
(a) PDB file of diffused CA trace
(b) TRB file for diffused CA trace

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

Two different flavors of structure prediction with AlphaFold2 were used in this study. For most rounds of design, standard single-sequence prediction with model 4 was performed using the superfold wrapper (https://github.com/rdkibler/superfold). For the designs with N+1 oxyanion hole motifs and the 4MU-phenylacetate substrate, we used slightly modified version Initial Guess AlphaFold2 (https://www.nature.com/articles/s41467-023-38328-5). In this case, all-atom positions for a chosen number of long-range residue-residue contacts  were input in the template. This increased the fraction of designs that folded to the correct structure, but a majority of designs still did not fold.

For both flavors, the inputs are similar.
Inputs:
(a) PDB file of design model
(b) If using initial guess: number of contacts to input to template

Outputs:
(a) PDB file of AlphaFold2 prediction
(b) JSON file of prediction scores

### Step 05: Copy substrate from design into AlphaFold2 model

To prepare AlphaFold2 predictions for the ChemNet runscript, we need to add the substrate from the design model roughly into the active site. Using the above AlphaFold2 scripts, the predictions should already be aligned onto the design model, making this simply a question of copying the substrate coordinates from one file to another.

Inputs:
(a) PDB file of design model containing substrate
(b) PDB file of AlphaFold2 prediction

Outputs:
(a) PDB file of AlphaFold2 prediction containing substrate

### Step 06: Run ChemNet

We run ChemNet in each of 5 modes representing the 5 intermediates along the reaction coordinate (apo, substrate-bound, tetrahedral intermediate 1 (T1), acylenzyme intermediate (AEI), tetrahedral intermediate 2 (T2)).

Inputs:
(a) PDB file of AlphaFold2 prediction containing substrate
(b) Residue number & chain of catalytic serine and the 3-letter code for the intended serine modification (for acylenzyme or tetrahedral intermediates). For the 4MU-butyrate substrate: T1=Q6R, AEI=OAS, T2=ZCX.
(c) Number of models in output ensemble

Outputs:
(a) PDB file containing ChemNet ensemble
(b) CSV file containing additional metrics from ChemNet



