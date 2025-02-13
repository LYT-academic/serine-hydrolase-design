# Python standard library
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

# 3rd party library imports
import numpy as np
from pyrosetta.distributed import requires_init

# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Custom library imports


def dict_to_fasta(
    seqs_dict: Dict[str, str],
    out_path: str,
) -> None:
    """
    :param seqs_dict: dictionary of sequences to write to a fasta file.
    :param out_path: path to write the fasta file to.
    :return: None
    Write a fasta file to the provided path with the provided sequence dict.
    """
    import os
    from pathlib import Path

    # make the output path if it doesn't exist
    if not os.path.exists(Path(out_path).parent):
        os.makedirs(Path(out_path).parent)
    else:
        pass
    # write the sequences to a fasta file
    with open(out_path, "w") as f:
        for i, seq in seqs_dict.items():
            f.write(f">{i}\n{seq}\n")
    return


def fasta_to_dict(fasta: str, new_tags: bool = False) -> Dict[str, str]:
    """
    :param fasta: fasta filepath to read from.
    :param new_tags: if False, use the sequence tag as the key, if True, use the index.
    :return: dictionary of tags and sequences.
    Read in a fasta file and return a dictionary of tags and sequences.
    """
    seqs_dict = {}

    with open(fasta, "r") as f:
        i = 0
        for line in f:
            if line.startswith(">"):  # this is a header line
                if new_tags:
                    tag = str(i)
                else:
                    tag = line.strip().replace(">", "")
                seqs_dict[tag] = ""
                i += 1
            else:  # we can assume this is a sequence line, add the sequence
                seqs_dict[tag] += line.strip()

    return seqs_dict



@requires_init
def thread_full_sequence(
    pose: Pose,
    sequence: str,
    start_res: int = 1,
) -> Pose:
    """
    :param: pose: Pose to thread sequence onto.
    :param: sequence: Sequence to thread onto pose.
    :return: Pose with threaded sequence.
    Threads a full sequence onto a pose after cloning the input pose.
    Doesn't require the chainbreak '/' to be cleaned.
    """
    from pyrosetta.rosetta.protocols.simple_moves import SimpleThreadingMover

    try:
        assert sequence.count("/") == 0
    except AssertionError:
        sequence = sequence.replace("/", "")

    pose = pose.clone()
    stm = SimpleThreadingMover()
    stm.set_sequence(sequence, start_res)
    stm.apply(pose)

    return pose


class MPNNRunner(ABC):
    """
    Abstract base class for MPNN runners.
    """

    import os
    import pwd
    import shutil
    import subprocess
    import uuid

    import pyrosetta.distributed.io as io

    def __init__(
        self,
        batch_size: Optional[int] = 8,
        model_name: Optional[str] = "v_48_030",
        path_to_model_weights: Optional[str] = "/databases/mpnn/vanilla_model_weights/",
        num_sequences: Optional[int] = 64,
        omit_AAs: Optional[str] = "X",
        temperature: Optional[float] = 0.1,
        backbone_noise: Optional[float] = 0.0,
        design_selector: Optional[ResidueSelector] = None,
        chains_to_mask: Optional[List[str]] = None,
        mpnn_subtype: Optional[str] = "vanilla",
        pack_side_chains: Optional[bool] = False,
        params: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the base class for MPNN runners with common attributes.
        :param: batch_size: number of sequences to generate per batch.
        :param: model_name: name of the model to use. v_48_030 is probably best in most
        cases where there may be imperfections in the backbone, v_32* use less memory.
        :param: path_to_model_weights: absolute path to model weights. Change if running
        somewhere besides the digs.
        :param: num_sequences: number of sequences to generate in total.
        :param: omit_AAs: concatenated string of amino acids to omit from the sequence.
        :param: temperature: temperature to use for the MPNN.
        :param: design_selector: ResidueSelector that specifies residues to design.
        :param: chains_to_mask: list of chains to mask, these will be designable.
        :param: mpnn_subtype: type of MPNN to use.
        :param: pack_side_chains: if True, pack sidechains after design.
        :param: params: path to params file to use for LigandMPNN.
        If no `chains_to_mask` is provided, the runner will run on (mask) all chains.
        If a `chains_to_mask` is provided, the runner will run on (mask) only that chain.
        If no `design_selector` is provided, all residues on all masked chains will be designed.
        The chain letters in your PDB must be correct.
        No mcmc/AF2 stuff currently implemented.
        """

        from pathlib import Path

        self.batch_size = batch_size
        self.model_name = model_name
        self.path_to_model_weights = path_to_model_weights
        self.num_sequences = num_sequences
        self.omit_AAs = omit_AAs
        self.temperature = temperature
        self.backbone_noise = backbone_noise
        self.design_selector = design_selector
        self.chains_to_mask = chains_to_mask
        try:
            assert mpnn_subtype in ["vanilla", "ligandMPNN"]
        except AssertionError:
            raise NotImplementedError("Only vanilla and ligandMPNN are implemented.")
        self.mpnn_subtype = mpnn_subtype
        self.pack_side_chains = int(pack_side_chains)
        self.params = params
        # setup standard command line flags for MPNN with default values
        self.flags = {
            "--max_length": "20000",
        }
        # add the flags that are required by MPNN and provided by the user
        self.flags.update(
            {
                "--batch_size": str(self.batch_size),
                "--num_seq_per_target": str(self.num_sequences),
                "--model_name": self.model_name,
                "--path_to_model_weights": self.path_to_model_weights,
                "--omit_AAs": self.omit_AAs,
                "--sampling_temp": str(self.temperature),
                "--backbone_noise": str(self.backbone_noise),
                "--pack_side_chains": str(self.pack_side_chains),
            }
        )
        self.allowed_flags = set(
            [
                # flags that have default values that are provided by the runner:
                "--backbone_noise",
                "--max_length",
                # flags that are set by MPNNRunner constructor:
                "--batch_size",
                "--num_seq_per_target",
                "--pack_side_chains",
                "--path_to_model_weights",
                "--model_name",
                "--omit_AAs",
                "--sampling_temp",
                # flags that are required and are set by MPNNRunner or children:
                "--jsonl_path",
                "--chain_id_jsonl",
                "--fixed_positions_jsonl",
                "--out_folder",
                # flags that are optional and are set by MPNNRunner or children:
                "--assume_symmetry",
                "--bias_AA_jsonl",
                "--bias_by_res_jsonl",
                "--compute_input_sequence_score",
                "--omit_AA_jsonl",
                "--tied_positions_jsonl",
                "--pdb_path",
                "--pdb_path_chains",
                "--pssm_bias_flag",
                "--pssm_jsonl",
                "--pssm_log_odds_flag",
                "--pssm_multi",
                "--pssm_threshold",
                "--save_probs",
                "--save_score",
                "--seed",
                "--use_seed",
                "--conditional_probs_only",
                "--conditional_probs_only_backbone",
                "--conditional_probs_use_pseudo",
                "--unconditional_probs_only",
                "--score_sc_only",
                "--pack_only",
                "--num_packs",
                "--pdb_bias_path",
                "--pdb_bias_level",
                "--species",
                "--transmembrane",
                "--transmembrane_buried",
                "--transmembrane_interface",
                "--transmembrane_chain_ids",

            ]
        )
        if self.mpnn_subtype == "vanilla":
            self.script = f"{str(Path(__file__).resolve().parent.parent.parent / 'proteinmpnn' / 'protein_mpnn_run.py')}"
        else:
            self.script = f"{str(Path(__file__).resolve().parent.parent.parent / 'proteinmpnn' / self.mpnn_subtype / 'protein_mpnn_run.py')}"
            #self.flags.pop("--path_to_model_weights")
            #self.flags.pop("--model_name")
            #self.allowed_flags.remove("--path_to_model_weights")
            #self.allowed_flags.remove("--model_name")
        self.tmpdir = None  # this will be updated by the setup_tmpdir method.
        self.is_setup = False  # this will be updated by the setup_runner method.

    def get_flags(self) -> Dict[str, str]:
        """
        :return: dictionary of flags.
        """
        return self.flags

    def get_script(self) -> str:
        """
        :return: script path.
        """
        return self.script

    def setup_tmpdir(self) -> None:
        """
        :return: None
        Create a temporary directory for the MPNNRunner. Checks for various best
        practice locations for the tmpdir in the following order: TMPDIR, PSCRATCH,
        CSCRATCH, /net/scratch. Uses the cwd if none of these are available.
        """
        import os
        import pwd
        import uuid

        if os.environ.get("TMPDIR") is not None:
            tmpdir_root = os.environ.get("TMPDIR")
        elif os.environ.get("PSCRATCH") is not None:
            tmpdir_root = os.environ.get("PSCRATCH")
        elif os.environ.get("CSCRATCH") is not None:
            tmpdir_root = os.environ.get("CSCRATCH")
        elif os.path.exists("/net/scratch"):
            tmpdir_root = f"/net/scratch/{pwd.getpwuid(os.getuid()).pw_name}"
        else:
            tmpdir_root = os.getcwd()

        self.tmpdir = os.path.join(tmpdir_root, uuid.uuid4().hex)
        os.makedirs(self.tmpdir, exist_ok=True)
        return

    def teardown_tmpdir(self) -> None:
        """
        :return: None
        Remove the temporary directory for the MPNNRunner.
        """
        import shutil

        if self.tmpdir is not None:
            try:
                shutil.rmtree(self.tmpdir)
            except OSError as e:
                print(f"Failed to remove tmpdir {self.tmpdir} with error {e}")
        return

    def update_flags(self, update_dict: Dict[str, str]) -> None:
        """
        :param: update_dict: dictionary of flags to update.
        :return: None
        Update the flags dictionary with the provided dictionary.
        Validate the flags before updating.
        """

        for flag in update_dict.keys():
            if flag not in self.allowed_flags:
                raise ValueError(
                    f"Flag {flag} is not allowed. Allowed flags are {self.allowed_flags}"
                )
        self.flags.update(update_dict)
        return

    def setup_runner(self, pose: Pose) -> None:
        """
        :param: pose: Pose object to run MPNN on.
        :return: None
        Setup the MPNNRunner. Make a tmpdir and write input files to the tmpdir.
        Output sequences and scores will be written temporarily to the tmpdir as well.
        """
        import json
        import os
        import shutil
        import sys
        from pathlib import Path

        import git
        import pyrosetta
        import pyrosetta.distributed.io as io

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.utils.io import cmd

        # setup the tmpdir
        self.setup_tmpdir()
        # check that the first residue is on chain A, if not we should try to fix it
        if pose.pdb_info().chain(1) != "A":
            # ensure the chain numbers are correct by using SwitchChainOrderMover
            sc = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
            sc.chain_order("".join([str(c) for c in range(1, pose.num_chains() + 1)]))
            sc.apply(pose)
        else:
            pass
        # write the pose to a clean PDB file of only ATOM coordinates.
        tmp_pdb_path = os.path.join(self.tmpdir, "tmp.pdb")
        pdbstring = io.to_pdbstring(pose)
        with open(tmp_pdb_path, "w") as f:
            f.write(pdbstring)
        # make the jsonl file for the PDB biounits
        biounit_path = os.path.join(self.tmpdir, "biounits.jsonl")
        # use git to find the root of the repo
        repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
        root = repo.git.rev_parse("--show-toplevel")
        if os.environ.get('cs_python_path') is not None:
            python = os.environ.get('cs_python_path') 
       # elif os.path.exists(str(Path(root) / "envs" / "shifty" / "bin" / "python")):
       #     python = str(Path(root) / "envs" / "shifty" / "bin" / "python")
        elif 'APPTAINER_CONTAINER' in list(os.environ.keys()): #if already in an apptainer, use /usr/bin/python.  Is this safe?
            python = '/usr/bin/python'
        elif os.path.exists(str(Path(root) / "containers" / "shifty.sif")): #if shifty.sif is in the repo use that
            python = str(Path(root) / "containers" / "shifty.sif")
        else:  # use /software shifty.sif
            python = '/software/containers/shifty.sif'
        if self.mpnn_subtype == "vanilla":
            parser_script = str(
                Path(__file__).resolve().parent.parent.parent
                / "proteinmpnn"
                / "helper_scripts"
                / "parse_multiple_chains.py"
            )
        else:
            parser_script = str(
                Path(__file__).resolve().parent.parent.parent
                / "proteinmpnn"
                / self.mpnn_subtype
                / "helper_scripts"
                / "parse_multiple_chains.py"
            )
        if self.params is None:
            # continue if no ligand params file is provided
            pass
        else:
            # need to copy over the ligand params file to the tmpdir
            tmp_params_path = os.path.join(self.tmpdir, "tmp.params")
            shutil.copy(self.params, tmp_params_path)

        run_cmd = " ".join(
            [
                f"{python} {parser_script}",
                f"--input_path {self.tmpdir}",
                f"--output_path {biounit_path}",
            ]
        )
        out_err = cmd(run_cmd)
        print(out_err)
        # cmd_real_time(run_cmd)
        # make a number to letter dictionary that starts at 1
        num_to_letter = {
            i: chr(i - 1 + ord("A")) for i in range(1, pose.num_chains() + 1)
        }
        # make the jsonl file for the chain_ids
        chain_id_path = os.path.join(self.tmpdir, "chain_id.jsonl")
        chain_dict = {}
        # make lists of masked and visible chains
        masked, visible = [], []
        # first make a list of all chain letters in the pose
        all_chains = [num_to_letter[i] for i in range(1, pose.num_chains() + 1)]
        # if chains_to_mask is provided, update the masked and visible lists
        if self.chains_to_mask is not None:
            # loop over the chains in the pose and add them to the appropriate list
            for chain in all_chains:
                if chain in self.chains_to_mask:
                    masked.append(i)
                else:
                    visible.append(i)
        else:
            # if chains_to_mask is not provided, mask all chains
            masked = all_chains
        chain_dict["tmp"] = [masked, visible]
        # write the chain_dict to a jsonl file
        with open(chain_id_path, "w") as f:
            f.write(json.dumps(chain_dict))
        # make the jsonl file for the fixed_positions
        fixed_positions_path = os.path.join(self.tmpdir, "fixed_positions.jsonl")
        fixed_positions_dict = {"tmp": {chain: [] for chain in all_chains}}
        # get a boolean mask of the residues in the design_selector
        if self.design_selector is not None:
            designable_filter = list(self.design_selector.apply(pose))
        else:  # if no design_selector is provided, make all residues designable
            designable_filter = [True] * pose.size()
        # check the residue design_selector specifies designability across the entire pose
        try:
            assert len(designable_filter) == pose.total_residue()
        except AssertionError:
            print(
                "Residue selector must specify designability for all residues.\n",
                f"Selector: {len(list(self.design_selector.apply(pose)))}\n",
                f"Pose: {pose.size()}",
            )
            raise
        # make a dict mapping of residue numbers to whether they are designable
        designable_dict = dict(zip(range(1, pose.size() + 1), designable_filter))
        # loop over the chains and the residues in the pose
        i = 1  # total residue counter
        for chain_number, chain in enumerate(all_chains, start=1):
            j = 1  # chain residue counter
            for res in range(
                pose.chain_begin(chain_number), pose.chain_end(chain_number) + 1
            ):
                # if the residue is on a masked chain but not designable, add it
                if not designable_dict[i] and chain in masked:
                    fixed_positions_dict["tmp"][chain].append(j)
                else:
                    pass
                j += 1
                i += 1
        # write the fixed_positions_dict to a jsonl file
        with open(fixed_positions_path, "w") as f:
            f.write(json.dumps(fixed_positions_dict))
        # update the flags for the biounit, chain_id, and fixed_positions paths
        flag_update = {
            "--jsonl_path": biounit_path,
            "--chain_id_jsonl": chain_id_path,
            "--fixed_positions_jsonl": fixed_positions_path,
        }
        self.update_flags(flag_update)
        self.is_setup = True
        return

    @abstractmethod
    def apply(self) -> None:
        """
        This function needs to be implemented by the child class of MPNNRunner.
        """
        pass


class MPNNDesign(MPNNRunner):
    """
    Class for running MPNN on a single interface selection or chain.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        :param: args: arguments to pass to MPNNRunner.
        :param: kwargs: keyword arguments to pass to MPNNRunner.
        Initialize the base class for MPNN runners with common attributes.
        """
        super().__init__(*args, **kwargs)

    def apply(self, pose: Pose) -> None:
        """
        :param: pose: Pose object to run MPNN on.
        :return: None
        Run MPNN on the provided pose.
        Setup the MPNNRunner using the provided pose.
        Run MPNN in a subprocess using the provided flags and tmpdir.
        Read in and parse the output fasta file to get the sequences.
        Each sequence designed by MPNN is then appended to the pose datacache.
        Cleanup the tmpdir.
        """
        import os
        import socket
        import sys
        from pathlib import Path

        import git
        import pyrosetta
        import pyrosetta.distributed.io as io
        from pyrosetta.rosetta.core.pose import setPoseExtraScore

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.mpnn import fasta_to_dict
        from crispy_shifty.utils.io import cmd

        # setup runner
        self.setup_runner(pose)
        self.update_flags({"--out_folder": self.tmpdir})

        # run mpnn by calling self.script and providing the flags
        # use git to find the root of the repo
        repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
        root = repo.git.rev_parse("--show-toplevel")
        if os.environ.get('cs_python_path') is not None:
            python = os.environ.get('cs_python_path') 
       # elif os.path.exists(str(Path(root) / "envs" / "shifty" / "bin" / "python")):
       #     python = str(Path(root) / "envs" / "shifty" / "bin" / "python")
        elif 'APPTAINER_CONTAINER' in list(os.environ.keys()): #if already in an apptainer, use /usr/bin/python.  Is this safe?
            python = '/usr/bin/python'
        elif os.path.exists(str(Path(root) / "containers" / "shifty.sif")): #if shifty.sif is in the repo use that
            python = str(Path(root) / "containers" / "shifty.sif")
        else:  # use /software shifty.sif
            python = '/software/containers/shifty.sif'
        if self.mpnn_subtype == "ligandMPNN":
            # hacky way to get around confusing MPNN
            self.flags.pop("--chain_id_jsonl")
            self.flags.pop("--jsonl_path")
            self.update_flags({"--pdb_path": str(Path(self.tmpdir) / "tmp.pdb")})
            if self.params:
                self.update_flags(
                    {"--ligand_params_path": str(Path(self.tmpdir) / "tmp.params")}
                )
        else:
            pass
        run_cmd = (
            f"{python} {self.script}"
            + " "
            + " ".join([f"{k} {v}" for k, v in self.flags.items()])
        )
        print(
            f"Running MPNN at {self.tmpdir} on {socket.gethostname()} with the following command:"
        )
        print(run_cmd, flush=True)
        out_err = cmd(run_cmd)
        print(out_err)
        # cmd_real_time(run_cmd)
        alignments_path = os.path.join(self.tmpdir, "seqs/tmp.fa")
        # parse the alignments fasta into a dictionary
        alignments = fasta_to_dict(alignments_path, new_tags=False)
        # get MPNN metadata from the key in the dict that contains mpnn_seq_0000_tmp
        mpnn_metadata = [k for k in alignments.keys() if "tmp," in k][0]
        # get individual metadata from the tag
        mpnn_metadata_dict = {}
        mpnn_metadata_dict["seed"] = mpnn_metadata.split("seed=")[1].split(",")[0]
        mpnn_metadata_dict["model_name"] = mpnn_metadata.split("model_name=")[1].split(
            ","
        )[0]
        if "git_hash" in mpnn_metadata:
            mpnn_metadata_dict["git_hash"] = mpnn_metadata.split("git_hash=")[1].split(
                ","
            )[0]
        else:
            pass
        if "ligand_rmsd" in mpnn_metadata:
            mpnn_metadata_dict["ligand_rmsd"] = mpnn_metadata.split("ligand_rmsd=")[
                1
            ].split(",")[0]
        else:
            pass
        # add the metadata to the pose datacache along with the alignments
        for i, (tag, seq) in enumerate(alignments.items()):
            index = str(i).zfill(4)
            setPoseExtraScore(pose, f"mpnn_seq_{index}", seq)
            # get score and seq_recovery from the tag
            mpnn_score = float(tag.split("score=")[1].split(",")[0])
            if "tmp," in tag:  # native seq has "tmp" in the tag and 100% seq recovery
                mpnn_seq_recovery = 1.0
            else:
                mpnn_seq_recovery = float(tag.split("seq_recovery=")[1].split(",")[0])
            # set the mpnn_score and mpnn_seq_recovery in the pose scores
            setPoseExtraScore(pose, f"mpnn_score_{index}", mpnn_score)
            setPoseExtraScore(
                pose, f"mpnn_sequence_recovery_{index}", mpnn_seq_recovery
            )
        for k, v in mpnn_metadata_dict.items():
            setPoseExtraScore(pose, f"mpnn_{k}", v)
        # clean up the temporary files if we won't need to get at the pdbs
        if not self.pack_side_chains:
            self.teardown_tmpdir()
        else:
            pass
        return

    def dump_fasta(self, pose: Pose, out_path: str) -> None:
        """
        :param: pose: Pose object that contains the designed sequences.
        :param: out_path: Path to write the fasta file to.
        :return: None
        Dump the pose mpnn_seq_* sequences to a single fasta.
        """
        import sys
        from pathlib import Path

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.mpnn import dict_to_fasta

        # get the mpnn_seq_* sequences from the pose
        seqs_to_write = {
            tag: seq for tag, seq in pose.scores.items() if "mpnn_seq_" in tag
        }
        # write the sequences to a fasta
        dict_to_fasta(seqs_to_write, out_path)

        return

    def generate_all_poses(
        self, pose: Pose, include_native: bool = False
    ) -> Iterator[Pose]:
        """
        :param: pose: Pose object to generate poses from.
        :param: include_native: Whether to generate the original native sequence.
        :return: Iterator of Pose objects.
        Generate poses from the provided pose with the newly designed sequences.
        Maintain the scores of the provided pose in the new poses.
        Add mpnn metadata for score and seq_recovery
        NB only returns the first packed model if running sidechain packing.
        """
        import sys
        from pathlib import Path

        import pyrosetta
        from pyrosetta.rosetta.core.pose import setPoseExtraScore

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.mpnn import thread_full_sequence

        # get the mpnn_seq_* sequences from the pose
        seqs_dict = {
            tag: seq for tag, seq in pose.scores.items() if "mpnn_seq_" in tag
        }
        # get the non-mpnn_seq_* scores from the pose that aren't the individual mpnn scores
        scores_dict = {
            key: val
            for key, val in pose.scores.items()
            if (
                "mpnn_seq_" not in key
                and "mpnn_sequence_recovery_" not in key
                and "mpnn_score_" not in key
            )
        }
        # get the mpnn metadata from the pose
        mpnn_metadata_dict = {}
        for key, val in pose.scores.items():
            if "mpnn_sequence_recovery_" in key:
                mpnn_metadata_dict[key] = val
            elif "mpnn_score_" in key:
                mpnn_metadata_dict[key] = val
            else:
                pass
        # reset the pose scores datacache
        pyrosetta.rosetta.core.pose.clearPoseExtraScores(pose)
        # generate the poses from the seqs_dict
        try:
            for tag, seq in seqs_dict.items():
                index = tag.split("_")[-1]
                # check if it's the native sequence
                if "_0000" in tag:
                    # if we want to include the native sequence we need to get its score
                    if include_native:
                        threaded_pose = pose.clone()
                    else:
                        continue
                else:
                    # check if self.pack_side_chains is True, thread if not
                    if self.pack_side_chains:
                        # find and load the corresponding pdb for this sample
                        # mpnn_sample = tag.split("sample=")[1].split(",")[0]
                        mpnn_sample = int(index)-1
                        packed_pdb = str(
                            Path(self.tmpdir)
                            / f"packed/tmp_seq_{mpnn_sample}_packed_0.pdb"
                        )
                        # load the packed pdb
                        threaded_pose = pyrosetta.pose_from_file(packed_pdb)
                    else:
                        # thread the full sequence
                        threaded_pose = thread_full_sequence(pose, seq)
                # add the mpnn metadata to the scores dict (overwriting each loop)
                # look it up in the mpnn_metadata_dict
                scores_dict["mpnn_score"] = mpnn_metadata_dict[f"mpnn_score_{index}"]
                scores_dict["mpnn_sequence_recovery"] = mpnn_metadata_dict[
                    f"mpnn_sequence_recovery_{index}"
                ]
                # set the scores
                for key, val in scores_dict.items():
                    setPoseExtraScore(threaded_pose, key, val)
                yield threaded_pose
        finally:
            # don't forget to clean up the tmpdir!
            if self.pack_side_chains:
                self.teardown_tmpdir()
            else:
                pass


class MPNNLigandDesign(MPNNDesign):
    """
    Class for running MPNN on a ligand containing pose.
    """

    def __init__(
        self,
        *args,
        model_name: str = "v_32_010",
        path_to_model_weights: Optional[str] = "/databases/mpnn/ligand_model_weights/",
        checkpoint_path: str = None,
        params: str = None,
        **kwargs,
    ):
        """
        :param: args: arguments to pass to MPNNRunner.
        :param: params: Path to the params file to use.
        :param: kwargs: keyword arguments to pass to MPNNRunner.
        Initialize the base class for MPNN runners with common attributes.
        Update the flags to use the provided params file.
        Doesn't allow changing hidden_dim, num_layers, or num_connections.
        """
        # override the default model_name in the parent class
        super().__init__(
            *args,
            model_name=model_name,
            path_to_model_weights=path_to_model_weights,
            mpnn_subtype="ligandMPNN",
            checkpoint_path=checkpoint_path,
            params=params,
            **kwargs,
        )
        self.params = params
        # ensure the params file is provided
        # Hmm, not necessary for DNA though- removing for now
        # if self.params is None:
        #     raise ValueError("Must provide params file.")
        # else:
        #     pass
        # add allowed flags for the older inference script
        self.allowed_flags.update(
            [
                "--use_sc",
                "--use_DNA_RNA",
                "--use_ligand",
                "--mask_hydrogen",
                "--ligand_params_path",
                "--checkpoint_path",
                "--random_ligand_rotation",
                "--random_ligand_translation",
                "--checkpoint_path_sc",
                "--path_to_model_weights_sc",
                "--model_name_sc",
            ]
        )
        # take out model_name from the allowed flags
        self.allowed_flags.remove("--assume_symmetry")
        #self.allowed_flags.remove("--bias_by_res_jsonl")
        self.allowed_flags.remove("--compute_input_sequence_score")
        self.allowed_flags.remove("--use_seed")
        #self.allowed_flags.remove("--conditional_probs_only")
        #self.allowed_flags.remove("--conditional_probs_only_backbone")
        #self.allowed_flags.remove("--conditional_probs_use_pseudo")
        #self.allowed_flags.remove("--unconditional_probs_only")


