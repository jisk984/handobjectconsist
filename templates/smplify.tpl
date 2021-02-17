which conda
conda activate physimpy37
cd /gpfswork/rech/tan/usk19gv/code/pose_3d/handobjectconsist/
export TMPDIR=$SCRATCH
module load vtk/8.1.2-mpi
module load cmake/3.18.0

python obmanmeshprocess.py --data_step {data_step} --data_offset {data_offset}
