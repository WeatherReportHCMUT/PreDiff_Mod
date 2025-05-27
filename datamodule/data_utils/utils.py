
import os

from einops import rearrange
import numpy as np
import torch

def path_splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def change_layout(
    data,
    in_layout='NHWT', out_layout='NHWT',
    ret_contiguous=False
):
    in_layout = " ".join(in_layout.replace("C", "1"))
    out_layout = " ".join(out_layout.replace("C", "1"))
    data = rearrange(data, f"{in_layout} -> {out_layout}")
    if ret_contiguous:
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data)
        elif isinstance(data, torch.Tensor):
            data = data.contiguous()
        else:
            raise ValueError
    return data

if __name__ == "__main__":
    print(path_splitall("/data/sevir_lr/data/CATALOG.csv"))