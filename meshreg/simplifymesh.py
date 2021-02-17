#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pickle
import random
import string
import subprocess
import tempfile
import trimesh

from libyana.exputils import argutils


def random_string(string_len=10):
    gen_str = ''.join(
        random.choices(string.ascii_uppercase + string.digits, k=string_len))
    return gen_str


def random_filename(string_len=10, folder="tmp", ext=".obj"):
    filename = random_string(string_len)
    filepath = os.path.join(folder, f"{filename}{ext}")
    return filepath


def simplify_mesh(
    src_path,
    target_path=None,
    manifold_path="/gpfswork/rech/tan/usk19gv/ManifoldPlus/build/manifold",
    acvd_path="/gpfswork/rech/tan/usk19gv/code/ACVD/bin/ACVD",
    vert_nb=1000,
    tmp_folder="tmp",
    delete_tmp=True,
    verbose=True,
    save_pkl=True,
):
    # Make watertight mesh
    mani_path = random_filename(folder=tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)
    mani_args = [
        manifold_path, "--input", src_path, "--output", mani_path, "--depth",
        "8"
    ]
    if verbose:
        print("Generating watertight mesh")
        print(' '.join(mani_args))
    subprocess.run(mani_args)
    if not os.path.exists(manifold_path):
        raise ValueError(
            f"Couldn't create {mani_path} with command {' '.join(mani_args)}")

    # Resample mesh uniformly
    uni_path = random_filename(folder=tmp_folder)
    uni_name = random_string()
    uni_prefix = os.path.join(tmp_folder, uni_name)
    uni_args = [acvd_path, mani_path, str(vert_nb), "--", "-o", uni_prefix]
    subprocess.run(uni_args)
    unires_path = f"{uni_prefix}smooth_simplification.ply"
    if verbose:
        print("Generating uniform mesh")
        print(' '.join(uni_args))
    if not os.path.exists(unires_path):
        raise ValueError(
            f"Couldn't create {unires_path} with command {' '.join(uni_args)}")

    mesh = trimesh.load(unires_path, force="mesh")
    mesh.export(target_path)

    if save_pkl:
        with open(target_path.replace(".obj", ".pkl"), "wb") as p_f:
            pickle.dump(
                {
                    "vertices": np.array(mesh.vertices),
                    "faces": np.array(mesh.faces)
                }, p_f)

    if not os.path.exists(target_path):
        raise ValueError(f"Counldn't find target file {target_path}")

    tmp_files = [mani_path, unires_path]
    if delete_tmp:
        for tmp_file in tmp_files:
            os.remove(tmp_file)
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tmp.obj")
    parser.add_argument("--output", default="out.obj")
    args = parser.parse_args()
    argutils.print_args(args)

    simplify_mesh(args.input, target_path=args.output)
    print(f"Saved mesh to {args.output}")
