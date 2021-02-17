which conda
conda activate physimpy37
cd /gpfswork/rech/tan/usk19gv/code/pose_3d/handobjectconsist/
export TMPDIR=$SCRATCH

python trainmeshreg.py --center_jittering {center_jittering} --lr {lr} --mano_lambda_pose_reg 5e-05 --mano_lambda_recov_joints3d 0.5 --mano_lambda_shape 5e-5 --obj_lambda_recov_verts3d 0.5 --obj_scale_factor 0.0001 --obj_trans_factor 1000 --scale_jittering {scale_jittering} --train_datasets {train_dataset} --train_splits {train_splits} --val_dataset ho3dv2 --val_split {val_split} --batch_size {batch_size} --workers 10
