import os
import numpy as np
import torch
import h5py
from pathlib import Path
from typing import List, Optional, Set, Tuple
from torch.utils.data import Dataset, Subset
from torch.utils import data as torch_data
from glob import glob

import tqdm
import open3d as o3d

"""
data structure:
raw_data/
    name/
        train/ cmd.npy, xyz.npy, rgb.npy
        val/ cmd.npy, xyz.npy, rgb.npy
"""

def data_process(robot: str):
    pass

if  __name__ == "__main__":
    pass