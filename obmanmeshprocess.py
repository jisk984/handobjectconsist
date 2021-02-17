#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import pickle
from joblib import Parallel, delayed

from tqdm import tqdm

from meshreg import simplifymesh

pkl_path = "/gpfsstore/rech/tan/usk19gv/datasets/obmantrain.pkl"
vert_nb = 1000
parser = argparse.ArgumentParser()
parser.add_argument("--data_offset", type=int, default=0)
parser.add_argument("--data_step", type=int, default=1)
args = parser.parse_args()

with open(pkl_path, "rb") as p_f:
    obman_data = pickle.load(p_f)

obj_paths = sorted(list(set(
    obman_data['obj_paths'])))[args.data_offset::args.data_step]

srcs = [obj_path.replace(".pkl", ".obj") for obj_path in obj_paths]
tars = [obj_path.replace(".pkl", "_proc.obj") for obj_path in obj_paths]

obj_models = {}
for tar in tqdm(tars):
    with open(tar.replace(".obj", ".pkl"), "rb") as p_f:
        data = pickle.load(p_f)
        obj_models[tar] = data
import pudb
pu.db

# Parallel(n_jobs=10, verbose=8)(delayed(simplifymesh.simplify_mesh)(src, tar)
#                                for src, tar in zip(srcs, tars))
for obj_path in tqdm(obj_paths):
    tar_path = obj_path.replace(".pkl", "_proc.obj")
    if not os.path.exists(tar_path.replace(".obj", ".pkl")):
        simplifymesh.simplify_mesh(obj_path.replace(".pkl", ".obj"),
                                   tar_path,
                                   vert_nb=vert_nb)
