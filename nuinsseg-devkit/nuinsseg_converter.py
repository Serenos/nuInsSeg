import os
from unicodedata import category
import numpy as np
from nuinsseg import NuInsSeg
import argparse
import json

from nuscenes.utils import splits

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

# using cam and lidar data
CAM_SENSOR = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
POINT_SENSOR = 'LIDAR_TOP'

NAME_MAPPING = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
}

def split_scene(nusc, version):
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    scene_names = [s['name'] for s in nusc.scene]
    train_scenes = list(filter(lambda x: x in scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in scene_names, val_scenes))
    train_scenes = set([
        nusc.scene[scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        nusc.scene[scene_names.index(s)]['token']
        for s in val_scenes
    ])
    print('train_scenes: ', len(train_scenes))
    print('val_scenes: ', len(val_scenes))
    return train_scenes, val_scenes


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        default='datasets/nuscenes',
        help='specify the root path of dataset')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        required=False,
        help='specify the dataset version')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='datasets/nuinsseg/annotations',
        required=False,
        help='path to save the exported json')
    parser.add_argument(
        '--nproc',
        type=int,
        default=4,
        required=False,
        help='workers to process semantic masks')
    parser.add_argument('--extra-tag', type=str, default='nuinsseg')
    parser.add_argument('--split', type=str, default='val', required=False)
    args = parser.parse_args()
    return args

def get_img_annos(nuseg, img_info, cat2id):
    """Get instance segmentation map for an image.

    Args:
        nuseg (obj:`NuInsSeg`): NuInsSeg dataset object
        img_info (dict): Meta information of img

    Returns:
    """
    sd_token = img_info['token']
    image_id = img_info['id']
    #name_to_index = name_to_index_mapping(nuim.category)
    width, height = img_info['width'], img_info['height']
    # anns = [
    #     a for a in nuseg.sample_annotations if a['sample']
    # ]
    sd = nuseg.get('sample_data', sd_token)
    s = nuseg.get('sample', sd['sample_token'])

    anns = []
    for ann_token in s['anns']:
        if (ann_token, sd_token) in nuseg._token2ind['nuinsseg'].keys():
            ann = nuseg.get_nuinsseg(ann_token, sd_token)
            anns.append(ann)

    coco_anns=[]
    for i, ann in enumerate(anns):
        cat_id = cat2id[NAME_MAPPING[ann['category_name']]]
        x_min, y_min, x_max, y_max = ann['bbox_corners']
        #The instance mask empty due to: 1.occulusion 2.mismatch
        if ann['instance_mask']['counts'] == 'PXn[1':
            continue
        coco_ann = dict(
            image_id = image_id,
            category_id = cat_id,
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min],
            area = (x_max - x_min) * (y_max - y_min),
            segmentation = ann['instance_mask'],
            iscrowd = 0
        )
        coco_anns.append(coco_ann)
    
    return coco_anns
    
def export_nuseg_to_coco(nuseg, data_root, out_dir, version, split):
    print('Process category information')
    categories = []
    categories = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    cat2id = {k_v['name']: k_v['id'] for k_v in categories}

    images = []
    print('Process image meta information...')
    train_scenes, val_scenes = split_scene(nuseg, version)
    scene = train_scenes if split == 'train' else val_scenes
    
    samples = [s for s in nuseg.sample if s['scene_token'] in scene]
    key_cam_sample_data_tokens = []
    for sample in samples:
        for cam in CAM_SENSOR:
            key_cam_sample_data_tokens.append(sample['data'][cam])
    #key_cam_sample_data = [x for x in nuseg.sample_data if x['is_key_frame']==1 and x['sensor_modality']=='camera']
    for idx, key_cam_sample_data_token in enumerate(key_cam_sample_data_tokens):
        key_cam_sample_data = nuseg.get('sample_data', key_cam_sample_data_token)
        images.append(
            dict(
                id=idx,
                token=key_cam_sample_data['token'],
                file_name=key_cam_sample_data['filename'],
                width=key_cam_sample_data['width'],
                height=key_cam_sample_data['height']))


    global process_img_anno
    def process_img_anno(img_info):
        single_img_annos = get_img_annos(nuseg, img_info, cat2id)
        return single_img_annos

    print('Process img annotations...')
    outputs = []
    for img_info in images:
        outputs.append(process_img_anno(img_info))
    
    annotations = []
    for single_img_annos in outputs:
        for img_anno in single_img_annos:
            img_anno.update(id=len(annotations))
            annotations.append(img_anno) 
    
    coco_format_json = dict(
        images=images, annotations=annotations, categories=categories)

    os.makedirs(out_dir, exist_ok=1)
    out_file = os.path.join(out_dir, f'nuinsseg_{nuseg.version}_{split}.json')
    print(f'Annotation dumped to {out_file}')
    with open(out_file, 'w') as f:
        json.dump(coco_format_json, f)


if __name__ == '__main__':
    args = parse_args()
    dataroot = os.path.join(os.environ['HOME'], args.data_root)
    outdir = os.path.join(os.environ['HOME'], args.out_dir)
    print(dataroot, outdir, args)
    nuseg = NuInsSeg(version=args.version, dataroot=dataroot, verbose=True)
    export_nuseg_to_coco(nuseg, dataroot, outdir, args.version, args.split)