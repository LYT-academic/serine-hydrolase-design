# Python standard library
import bz2
import collections
import json
import os
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, NoReturn, Optional, Tuple, Union

# 3rd party library imports
import pandas as pd
import pyrosetta.distributed.io as io
import toolz

# Rosetta library imports
from pyrosetta.distributed import requires_init
from pyrosetta.distributed.cluster.exceptions import OutputError
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector
from tqdm.auto import tqdm

# Custom library imports


def cmd(command: str = "", wait: bool = True) -> str:
    """
    :param: command: Command to run.
    :param: wait: Wait for command to finish.
    :return: stdout plus sterr.
    Run a command.
    """
    import subprocess

    p = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if wait:
        out = str(p.communicate()[0]) + str(p.communicate()[1])
        return out
    else:
        return


