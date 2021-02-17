import pickle

import numpy as np
import torch
import trimesh
from libyana.meshutils.meshio import fast_load_obj
from manopth import manolayer
from pytorch3d import ops as pt3dops
from transforms3d.axangles import mat2axangle
from tqdm import tqdm

from meshreg.datasets.queries import BaseQueries, get_trans_queries
from meshreg.datasets import manoutils
from libyana.meshutils import meshnorm
from PIL import Image
from functools import lru_cache


class ObMan:
    def __init__(
        self,
        root="local_data/datasets",
        pkl_path="/gpfsstore/rech/tan/usk19gv/datasets/obmantrain.pkl",
        split="train",
        mano_root="assets/mano",
        ref_idx=0,
        sample_idx=None,
        shapenet_root="/gpfsscratch/rech/tan/usk19gv/datasets/ShapeNetCore.v2",
    ):
        """
        Args:
            sample_idx: if not None, fallback on using only one hand-object configuration in
                the whole dataset
        """
        super().__init__()

        self.name = "obman"
        self.sample_idx = sample_idx
        self.cam_extr = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ]).astype(np.float32)

        self.camintr = np.array([[480., 0., 128.], [0., 480., 128.],
                                 [0., 0., 1.]]).astype(np.float32)
        with open(pkl_path, "rb") as p_f:
            self.obman_data = pickle.load(p_f)
            self._size = len(self.obman_data['obj_paths'])
        self.has_dist2strong = False
        self.object_models = {}

        self.layer = manolayer.ManoLayer(
            joint_rot_mode="axisang",
            mano_root=mano_root,
            center_idx=None,
            use_pca=False,
            flat_hand_mean=True,
        )
        self.ref_idx = ref_idx
        self.hand_faces = np.array(
            trimesh.load("assets/mano/closed_mano.obj",
                         process=False).faces)[:, ::-1].copy()
        self.joint_nb = 21
        self.cent_layer = manolayer.ManoLayer(
            joint_rot_mode="axisang",
            mano_root=mano_root,
            center_idx=ref_idx,
            flat_hand_mean=True,
            use_pca=False,
            side="right",
        )
        self.all_queries = [
            BaseQueries.IMAGE,
            BaseQueries.JOINTS2D,
            BaseQueries.JOINTS3D,
            BaseQueries.OBJVERTS2D,
            BaseQueries.OBJVERTS3D,
            BaseQueries.OBJFACES,
            BaseQueries.HANDVERTS2D,
            # BaseQueries.HANDVIS2D,
            BaseQueries.HANDVERTS3D,
            BaseQueries.OBJCANVERTS,
            BaseQueries.SIDE,
            BaseQueries.CAMINTR,
            BaseQueries.JOINTVIS,
        ]
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)

        obj_models = {}
        obj_paths = [
            obj_path.replace(".pkl", "_proc.pkl")
            for obj_path in self.obman_data["obj_paths"]
        ]
        print(f"Loading obman object models")
        for tar in tqdm(obj_paths):
            with open(tar, "rb") as p_f:
                data = pickle.load(p_f)
                obj_models[tar] = data
        self.obj_models = obj_models
        # Get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

    def get_frame_info(self, sample_id, load_img=True):
        if noise is None:
            hand_noise = None
            obj_noise = None
        else:
            hand_noise = noise["hand_noise"]
            obj_noise = noise["obj_noise"]
        obj_path = self.obman_data["obj_paths"][sample_id]
        hand_poses_ref = self.obman_data["hand_poses"][sample_id]
        hand_pcas = self.obman_data["hand_pcas"][sample_id]
        hand_poses = manoutils.aa_from_pca(torch.Tensor(
            hand_pcas[np.newaxis]))[0].numpy()
        obj_transform = self.obman_data["obj_transforms"][sample_id]
        obj_scale = self.obman_data['meta_infos'][sample_id]['obj_scale']
        hand_verts3d = self.obman_data["hand_verts3d"][sample_id]

        hand_verts = (
            self.cent_layer(
                torch.Tensor(
                    np.concatenate([
                        np.array([0, 0, 0]),
                        hand_poses  # hand_pcas
                    ])).unsqueeze(0))[0] / 1000)
        sim_vals = pt3dops.corresponding_points_alignment(
            hand_verts,
            torch.Tensor(hand_verts3d).unsqueeze(0))
        rot = sim_vals.R[0]
        hand_trans = sim_vals.T[0]

        ax, angle = mat2axangle(rot.T)
        axangle = angle * ax
        hand_rots = np.concatenate([axangle, hand_poses])
        hand_shape = np.zeros(10)
        hand_info = dict(
            verts3d=hand_verts3d,
            faces=self.hand_faces,
            pca=hand_pcas,
            shape=np.zeros(10),
            trans=hand_trans.numpy(),
            rots=torch.Tensor(hand_rots),
            label="right_hand",
        )

        # Process object
        with open(obj_path.replace(".pkl", "_proc.pkl"), "rb") as p_f:
            mesh = pickle.load(p_f)
        obj_verts = mesh['vertices']
        obj_faces = mesh['faces']

        obj_rot = obj_transform[:3, :3].T
        obj_trans = obj_transform[:3, 3]
        # Jitter object translation and rotation
        trans_obj_verts = (obj_verts.dot(obj_rot) + obj_trans)
        obj_info = dict(
            verts3d=trans_obj_verts,
            verts3d_vhacd=trans_obj_verts,
            faces=obj_faces,
            faces_vhacd=obj_faces,
            path=obj_path,
        )
        camera = dict(
            resolution=(256, 256),
            TWC=torch.eye(4).float(),
            K=camintr.astype(np.float32),
            focal_nc=1,
        )
        return {"camera": camera, "hand_info": hand_info, "obj_info": obj_info}

    def get_image(self, idx):
        image_path = self.obman_data["image_names"][idx]
        img = Image.open(image_path).convert("RGB")
        return img

    @lru_cache(maxsize=512)
    def get_objmesh(self, idx):
        obj_path = self.obman_data["obj_paths"][idx]
        mesh = self.obj_models[obj_path.replace(".pkl", "_proc.pkl")]
        vertices = mesh['vertices']
        faces = mesh['faces']
        return vertices, faces

    def get_obj_faces(self, idx):
        vertices, faces = self.get_objmesh(idx)
        faces = np.array(faces).astype(np.int16)
        return faces

    def get_obj_verts_trans(self, idx):
        vertices, faces = self.get_objmesh(idx)
        obj_transform = self.obman_data["obj_transforms"][idx]
        hom_vertices = np.concatenate(
            [vertices, np.ones([vertices.shape[0], 1])], axis=1)
        trans_vertices = obj_transform.dot(hom_vertices.T).T[:, :3]
        trans_vertices = (self.cam_extr[:3, :3].dot(
            trans_vertices.transpose()).transpose())
        return trans_vertices.astype(np.float32)

    def get_obj_verts_can(self, idx, rescale=True, no_center=False):
        verts, _ = self.get_objmesh(idx)
        if rescale:
            return meshnorm.center_vert_bbox(verts, scale=False)
        elif no_center:
            return verts, np.array([0, 0]), 1
        else:
            return meshnorm.center_vert_bbox(verts, scale=False)
        return vertices.astype(np.float32)

    def get_objverts2d(self, idx):
        vertices = self.get_obj_verts_trans(idx)
        verts2d = self.project(vertices)
        return verts2d.astype(np.float32)

    def project(self, points3d):
        hom_2d = np.array(self.camintr).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return points2d.astype(np.float32)

    def get_hand_verts3d(self, idx):
        return self.obman_data['hand_verts3d'][idx].astype(np.float32)

    def get_hand_verts2d(self, idx):
        verts3d = self.get_hand_verts3d(idx)
        verts2d = self.project(verts3d)
        return verts2d.astype(np.float32)

    def get_camintr(self, idx):
        return self.camintr.astype(np.float32)

    def get_joints3d(self, idx):
        return self.obman_data['joints3d'][idx].astype(np.float32)

    def get_joints2d(self, idx):
        return self.obman_data['joints2d'][idx].astype(np.float32)

    def get_sample(self, idx):
        # Override idx if using a single instance in the dataset !
        if self.sample_idx is not None:
            idx = self.sample_idx
        frame_info = self.get_frame_info(idx)
        obs = dict(img=np.zeros((4, 4, 3)),
                   hands=[frame_info["hand_info"]],
                   objects=[frame_info["obj_info"]],
                   camera=frame_info["camera"],
                   sample_id=idx)
        return obs

    def __getitem__(self, idx):
        obs = self.get_sample(idx)
        return obs

    def get_sides(self, idx):
        return "right"

    def get_center_scale(self, idx):
        center = np.array([128, 128])
        scale = 200
        return center, scale

    def get_jointvis(self, idx):
        return np.ones(self.joint_nb)

    def __len__(self):
        return self._size
