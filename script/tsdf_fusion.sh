# !/usr/bin/env bash

ROOT_DIR="./outputs/test"

python tsdf_fusion.py --n_view 3 --voxel_size 1.5 \
--root_dir=$ROOT_DIR $@

# ROOT_DIR="./outputs_view30"

# python tsdf_fusion.py --n_view 30 --voxel_size 1.5 \
# --root_dir=$ROOT_DIR $@