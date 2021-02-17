#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

from manopth import manolayer
from libyana.verify.checkshape import check_shape

mano_layer = manolayer.ManoLayer(
    joint_rot_mode="axisang",
    mano_root="assets/mano",
    flat_hand_mean=False,
    use_pca=True,
)


def pca_from_aa(hand_pose, rem_mean=False):
    check_shape(hand_pose, (-1, 45), "hand_pose")
    inv_pca_mat = torch.inverse(mano_layer.th_comps.T)
    bsz = hand_pose.shape[0]
    inv_pca_mat_b = inv_pca_mat.unsqueeze(0).repeat(bsz, 1, 1)
    # Go back to hand reference pose (flat hand)
    if rem_mean:
        hand_pose_ref = hand_pose - mano_layer.th_hands_mean
    else:
        hand_pose_ref = hand_pose
    pca_pose = torch.bmm(inv_pca_mat_b, hand_pose_ref.unsqueeze(2))[:, :, 0]
    return pca_pose


def aa_from_pca(pca_pose, add_mean=False):
    check_shape(pca_pose, (-1, -1), "pca_pose")
    ncomps = pca_pose.shape[1]
    bsz = pca_pose.shape[0]
    pca_mat = mano_layer.th_comps.T.unsqueeze(0).repeat(bsz, 1, 1)

    hand_pose_ref = torch.bmm(pca_mat[:, :, :ncomps],
                              pca_pose.unsqueeze(2))[:, :, 0]
    if add_mean:
        hand_pose = hand_pose_ref + mano_layer.th_hands_mean
    else:
        hand_pose = hand_pose_ref
    return hand_pose
