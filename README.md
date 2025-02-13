# serine-hydrolase-design

Code repository for de novo design of serine hydrolases, to accompany the manuscript:

Lauko A, Pellock SJ, Sumida KH. 2025. Computational design of serine hydrolases. *Science* doi: [10.1101/2024.08.29.610411](https://doi.org/10.1126/science.adu2454)

Installing [Apptainer](https://apptainer.org/docs/admin/main/installation.html) will allow you to run all code using the pre-packaged python environments found in `containers/`.

Our manuscript describes the de novo design of serine hydrolases, which we performed in two phases: motif generation and design. This code repository is organized such that each substep of motif generation and design (described below) can be run individually with example inputs and outputs found in the `inputs/` and `outputs/` subdirectories. To run, navigate into the numbered folder within `motif_gen/` or `design_pipeline/` and run the command(s) in `cmd.sh`.

## Requirements:  
(a) CA RFdiffusion  
(b) [LigandMPNN](https://github.com/dauparas/LigandMPNN)  
(c) [Superfold](https://github.com/rdkibler/superfold) or [Initial Guess AF2](https://github.com/nrbennet/dl_binder_design)  
(d) [PLACER](https://github.com/baker-laboratory/PLACER)  

## A. Motif Generation
RFdiffusion takes as input motifs consisting of "stubs" that hold the catalytic residues. To avoid constraining design to existing serine hydrolase active sites, we generate our own.

### Step 01: Sampling histidine stubs.

The script `invrotzyme.py`, described [here](https://github.com/ikalvet/invrotzyme) samples the interactions between the substrate, serine motif, and histidine motif according to the provided constraint file, producing an array of serine hydrolase active sites. In this case, high probability histidine rotamers on 3-residue helical stubs are being sampled in hydrogen bonding geometries to the serine.

>__Inputs__:  
(a) Rosetta params file describing the substrate in a near-tetrahedral geometry.  
(b) PDB file containing the local structure around serine nucleophile from a natural hydrolase (1lns).  
(c) Rosetta enzyme constraint file describing the geometry of the serine-substrate and the serine-histidine interactions.  

>__Outputs__:  
(a) PDB files containing serine hydrolase active sites with substrate, serine, and histidine.  

### Step 02: Increasing motif complexity by diffusion

To add additional residues to the motifs (ex. Asp/Glu triad residue, sidechain oxyanion hole residue), we use RFdiffusion with simple Ser-His motifs generated in Step 01 to create new backbones. We then refine the outputs, and proceed to search the generated backbones for positions that can accomodate the aforementioned catalytic motifs.

### Step 02:
RFdiffusion to generate new backbones around a simple motif.

>__Inputs__:  
(a) PDB file produced in Step 01 containing substrate, catalytic serine, and histidine.  
(b) JSON file containing arguments for RFdiffusion. For further details see Methods, Supplementary Information, and [Watson et al.](https://www.nature.com/articles/s41586-023-06415-8)  

>__Outputs__:  
(a) PDB file of diffusion backbone CA trace  
(b) TRB file containing mapping of input motif sequence positions to diffusion output sequence positions and input arguments (needed for refinement)  

### Step 03: Refinement

The backbones generated in step 1 are only CA traces. Refinement generates all-atom models from the CA trace which can then be input to LigandMPNN and Rosetta FastRelax. Refinement uses the RFdiffusion runscript in a different mode, and outputs the refined PDB files into the same directory as the CA traces.

>__Inputs__:  
(a) PDB file of CA trace produced by RFdiffusion in Step 02.  
(b) TRB file generated in Step 02  

>__Outputs__:  
(a) PDB file of refined diffusion output.  

### Step 04: Search for positions that accomodate new catalytic elements

We search a backbone for positions from which a donor sidechain can make an H-bond with atom O1 (oxyanion) in the substrate (command 1), OR for positions from which an acceptor Asp/Glu can make an H-bond with atom ND1 of the catalytic histidine (command 2). We ensure that these positions are not overly close to other catalytic residues in sequence and are on secondary structure elements. Any successful matches become motifs for further backbone generation, thus successively building up active site complexity. 

>__Inputs__:  
(a) Refined diffusion output  
(b) Constraint files that define the geometry of H-bond interactions between each donor/acceptor sidechain  

>__Outputs__:  
(a) PDB files containing the input PDB file with the desired acceptor/donor residue mutated in a position that can accomodate an H-bond to the target atom. If no position was found, nothing is output.  

## B. Design Pipeline

With motif(s) in hand, we move on to design, which consists of RFdiffusion starting from the motif, refinement of the diffusion outputs, sequence design with LigandMPNN+FastRelax, structural validation with AlphaFold2, multi-step PLACER prediction, and analysis of PLACER outputs.

### Step 01: CA RFdiffusion

New backbones are generated that contain the input active site.

>__Inputs__:  
(a) PDB file of input motif  
(b) JSON file containing arguments for RFdiffusion. For further details see Methods, Supplementary Information and [Watson et al.](https://www.nature.com/articles/s41586-023-06415-8)  

>__Outputs__:  
(a) PDB file of diffused CA trace and motif  
(b) TRB file containing mapping of input motif sequence positions to diffusion output sequence positions and input arguments (needed for refinement)  

### Step 02: Refinement

The backbones generated in step 1 are only CA traces. Refinement generates all-atom models from the CA trace which can then be input to LigandMPNN and Rosetta. Refinement uses the RFdiffusion runscript in a different mode, and outputs the refined PDB files into the same directory as the CA traces.

>__Inputs__:  
(a) PDB file of diffused CA trace (from Step 01)  
(b) TRB file for diffused CA trace (from Step 01)  

>__Outputs__:  
(a) PDB file of refined all-atom RFdiffusion output  

### Step 03: Design with LigandMPNN+FastRelax

The sequences of RFdiffusion outputs are designed with LigandMPNN, then relaxed using Rosetta with enzyme constraints to hold the active site in place. The process is repeated three times.

>__Inputs__:  
(a) PDB file of refinement output, or an existing design model if performing sequence resampling  
(b) Rosetta enzyme constraint file specifying desired interaction geometries in active site  
(c) Rosetta params file of substrate  
(d) Locations of catalytic residues in sequence  
(e) Additional arguments described in scripts/design.py  

>__Outputs__:  
(a) PDB file of design output  
(b) CSV file containing design scores  

### Step 04: Structural validation with AlphaFold2

Two different flavors of structure prediction with AlphaFold2 were used in this study. For most rounds of design, standard single-sequence prediction with model 4 was performed using the `superfold` [wrapper](https://github.com/rdkibler/superfold). For the designs with N+1 oxyanion hole motifs and the 4MU-phenylacetate substrate, we used a slightly modified version [Initial Guess AlphaFold2](https://www.nature.com/articles/s41467-023-38328-5). In this case, all-atom positions for a chosen number of long-range residue-residue contacts were input in the template. This increased the fraction of designs that folded to the correct structure, but a majority of designs still did not fold.

For both flavors, the inputs are similar.
>__Inputs__:  
(a) PDB file of design model  
(b) If using initial guess: number of contacts to input to template  

>__Outputs__:  
(a) PDB file of AlphaFold2 prediction  
(b) JSON or CSV file of prediction scores  

### Step 05: Copy substrate from design into AlphaFold2 model

To prepare AlphaFold2 predictions for the PLACER runscript, we need to add the substrate from the design model roughly into the active site. Using the above AlphaFold2 scripts, the predictions should already be aligned onto the design model, making this simply a question of copying the substrate coordinates from one file to another.

>__Inputs__:  
(a) PDB file of design model containing substrate  
(b) PDB file of AlphaFold2 prediction  

>__Outputs__:  
(a) PDB file of AlphaFold2 prediction containing substrate  

### Step 06: Run PLACER

We run PLACER in each of 5 modes representing the 5 intermediates along the reaction coordinate (apo, substrate-bound, tetrahedral intermediate 1 (T1), acylenzyme intermediate (AEI), tetrahedral intermediate 2 (T2)).

>__Inputs__:  
(a) PDB file of AlphaFold2 prediction containing substrate  
(b) Residue number & chain of catalytic serine and the 3-letter code for the intended serine modification (for acylenzyme or tetrahedral intermediates). For example, for re-face attack on the 4MU-butyrate substrate: T1=`Q6R`, AEI=`OAS`, T2=`ZCX`.  
(c) Number of models in output ensemble  

>__Outputs__:  
(a) PDB file containing PLACER ensemble  
(b) CSV file containing additional metrics from PLACER  

Table of PLACER 3-letter modifications of Ser for each substrate, each face of attack (*re*/*si*)
|          |   TI1   |   AEI   |   TI2   |
| -------- | ------- | ------- | ------- |
| 4MU-Ac   | `Q6R`/`LB0` |   `OAS`   | `ZCX`/`QFR` |
| 4MU-Bu   | `899`/    |   `432`   | `75I`/    |
| 4MU-PhAc | `SW6`/`ZW1` |   `999`   | `9UI`/`RKB` |

### Step 07: Analyze PLACER outputs

The PLACER ensemble output in Step 06 is analyzed for interaction geometries. Interactions are evaluated for the presence of H-bonds. The analysis results in a CSV file containing all metrics from each prediction in the ensemble. The script used for this step is highly tailored specifically for analysis of serine hydrolase interactions in PLACER outputs, but the code can be repurposed to evaluate H-bonds or catalytic residue conformations for a different enzyme.

>__Inputs__:  
(a) PDB file containing PLACER ensemble (Step 06)  
(b) CSV file containing additional metrics from PLACER (Step 06)  
(c) Catalytic intermediate being modeled in PLACER ensemble (flag: `--step`)  

>__Outputs__:  
(a) CSV file containing measured geometries, H-bonding info, and additional metrics for each prediction in ensemble  


Note: The inputs and outputs described above are the files needed to run the scripts. In some cases, additional arguments and options are provided in `cmd.sh`. For a complete description of additional arguments and options, run any script with the `--help` or `-h` flag.
