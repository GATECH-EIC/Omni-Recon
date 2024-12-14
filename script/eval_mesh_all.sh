#!/bin/bash

gpu=0
num_src_view=3
scan_list=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)


for scan_id in {0..14}
  do
    echo "processing scan: ${scan_list[$scan_id]}"

    bash script/eval_mesh.sh --gpu ${gpu} --num_src_view ${num_src_view} --load_ckpt checkpoints/omni_recon.pt \
    --volume_reso 144 --scan ${scan_list[$scan_id]}  > test_log/scan${scan_list[$scan_id]}.log

  done
