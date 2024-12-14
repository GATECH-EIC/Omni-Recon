import os
import numpy as np

root_dir = "/data/yfu314/neuray_data/"

gso_scene_names, gso_scene_names_400= [], []
if os.path.exists(f'{root_dir}/google_scanned_objects'):
    for fn in os.listdir(f'{root_dir}/google_scanned_objects'):
        if os.path.isdir(os.path.join(f'{root_dir}/google_scanned_objects',fn)):
            gso_scene_names.append(f'gso/{fn}/black_raw')
            gso_scene_names_400.append(f'gso/{fn}/black_400')

dtu_names=['birds','bricks','snowman','tools']
dtu_name2scan_id={'tools':'scan37', 'snowman':'scan69', 'bricks':'scan40', 'birds':'scan106'}
dtu_train_scene_names = []
dtu_test_scene_names_400 = []
dtu_test_scene_names_800 = []
dtu_test_scene_names_1600 = []
if os.path.exists(f'{root_dir}/dtu_train') and os.path.exists(f'{root_dir}/dtu_test'):
    fns = os.listdir(f'{root_dir}/dtu_train')
    fns = [fn for fn in fns if fn.startswith('scan')]
    test_scenes = os.listdir(f'{root_dir}/dtu_test')

    test_scans = np.loadtxt('src/data_pretrain/dtu_test_scans.txt',dtype=np.str).tolist()
    train_scans = [fn for fn in fns if fn not in test_scans]
    dtu_train_scene_names = [f'dtu_train/{fn}' for fn in train_scans]
    dtu_test_scene_names_400 = [f'dtu_test/{fn}/black_400' for fn in test_scenes]
    dtu_test_scene_names_800 = [f'dtu_test/{fn}/black_800' for fn in test_scenes]
    dtu_test_scene_names_1600 = [f'dtu_test/{fn}/black_1600' for fn in test_scenes]

real_iconic_scene_names_8 = []
real_iconic_scene_names_4 = []
if os.path.exists(f'{root_dir}/real_iconic_noface'):
    fns = os.listdir(f'{root_dir}/real_iconic_noface')
    real_iconic_scene_names_8 = [f'real_iconic/{fn}/8' for fn in fns]
    real_iconic_scene_names_4 = [f'real_iconic/{fn}/4' for fn in fns]

space_scene_names = []
if os.path.exists(f'{root_dir}/spaces_dataset'):
    fns = os.listdir(f'{root_dir}/spaces_dataset/data/800')
    space_scene_names = [f'space/{fn}' for fn in fns]

real_estate_scene_names = []
if os.path.exists(f'{root_dir}/real_estate_dataset'):
    fns = os.listdir(f'{root_dir}/real_estate_dataset/train/frames')
    real_estate_scene_names = [f'real_estate/{fn}/450_800' for fn in fns]

nerf_syn_val_ids=['val-r_39', 'val-r_2', 'val-r_94', 'val-r_62', 'val-r_23', 'val-r_36']
nerf_syn_names = ['chair','drums','ficus','hotdog','lego','materials','mic','ship']

llff_names = ['fern','flower','fortress','horns','leaves','orchids','room','trex']
LLFF_ROOT = f'{root_dir}/llff_colmap'

NERF_SYN_ROOT = f'{root_dir}/nerf_synthetic'

replica_scene_names = ["replica/room_0", "replica/room_1", "replica/room_2", "replica/office_0",
                       "replica/office_1", "replica/office_2", "replica/office_3", "replica/office_4"]

scannet_scene_names = []
with open("src/data_pretrain/scannet_scenes.txt", "r") as f:
    scenes = f.readlines()
    for scene in scenes:
        scannet_scene_names.append(scene[:-11])