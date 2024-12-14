#!/bin/bash

# Default values
gpu=7
test_n_view=3
num_src_view=3
volume_reso=144
scan="65"
set=0
dir_suffix=""
trans_depth=2
voxel_size=1.5

load_ckpt=""
auto_ckpt=""

predict_weight=""

extract_geometry_all_views=""
src_via_dist=""

camera_correction=""
simple_appear_feat=""
tiny_shader=""

no_quant_eval=""


# Process command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) gpu="$2"; shift ;;
        --test_n_view) test_n_view="$2"; shift ;;
        --volume_reso) volume_reso="$2"; shift ;;
        --num_src_view) num_src_view="$2"; shift ;;
        --load_ckpt) load_ckpt="$2"; shift ;;
        --set) set="$2"; shift ;;
        --scan) scan="$2"; shift ;;
        --dir_suffix) dir_suffix="$2"; shift ;;
        --trans_depth) trans_depth="$2"; shift ;;
        --voxel_size) voxel_size="$2"; shift ;;
        --auto_ckpt) auto_ckpt="--auto_ckpt";; 
        --predict_weight) predict_weight="--predict_weight";;
        --camera_correction) camera_correction="--camera_correction";;
        --simple_appear_feat) simple_appear_feat="--simple_appear_feat";;
        --tiny_shader) tiny_shader="--tiny_shader";;
        --extract_geometry_all_views) extract_geometry_all_views="--extract_geometry_all_views";;
        --src_via_dist) src_via_dist="--src_via_dist";;
        --no_quant_eval) no_quant_eval="true";;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

ckpt_name='omni_recon'

output_dir="./outputs/mesh_results/${ckpt_name}_scan${scan}_set${set}${dir_suffix}"

DTU_PATH="/data/yfu314/dtu"

DTU_TEST="$DTU_PATH/DTU_TEST"
DTU_MVS_DATA="$DTU_PATH/SampleSet/MVS_Data/"

CUDA_VISIBLE_DEVICES=${gpu} python main.py --extract_geometry --set ${set} \
--test_n_view ${test_n_view} --num_src_view ${num_src_view} --test_ray_num 400 --volume_reso ${volume_reso} --test_dir ${DTU_TEST} \
--trans_depth ${trans_depth} --load_ckpt=${load_ckpt} --scan=${scan} --out_dir=${output_dir}  ${auto_ckpt} ${predict_weight}  ${extract_geometry_all_views} ${src_via_dist} ${camera_correction} ${simple_appear_feat} ${tiny_shader} 

CUDA_VISIBLE_DEVICES=${gpu} python evaluation/tsdf_fusion.py --n_view ${test_n_view} --voxel_size ${voxel_size} --root_dir=${output_dir} --scan=${scan}

mesh_dir="${output_dir}/mesh"

CUDA_VISIBLE_DEVICES=${gpu} python evaluation/clean_mesh.py --root_dir ${DTU_TEST} --n_view ${test_n_view} --set ${set} --out_dir ${mesh_dir} --scan=${scan}

if [ -z "${no_quant_eval}" ]; then
    CUDA_VISIBLE_DEVICES=${gpu} python evaluation/dtu_eval.py --dataset_dir ${DTU_MVS_DATA} --outdir ${mesh_dir} --scan=${scan}
fi