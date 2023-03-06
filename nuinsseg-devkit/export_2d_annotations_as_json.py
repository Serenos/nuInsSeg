# nuScenes dev-kit.
# Code written by Lixiang, 2022.

"""
Export 2D annotations(innstance masks and 2D box) to a .json file.

Note: Projecting tight 3d boxes to 2d generally leads to non-tight boxes.
      Furthermore it is non-trivial to determine whether a box falls into the image, rather than behind or around it.
      Finally some of the objects may be occluded by other objects, in particular when the lidar can see them, but the
      cameras cannot.
"""

import argparse
import json
import os
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from shapely.geometry import MultiPoint, box
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def get_2d_instance_mask(anno_2d: List[OrderedDict]) -> List[OrderedDict]:
    s_rec = nusc.get('sample', anno_2d['sample_token'])
    sd_rec_token = s_rec['data'][anno_2d['cam']]
    sd_rec = nusc.get('sample_data', sd_rec_token)


    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')

    instance_rec = OrderedDict()
    instance_rec['sample_data_token'] = sd_rec_token
    ann_rec = nusc.get('sample_annotation', anno_2d['ann_token'])
    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            instance_rec[key] = value

    instance_rec['bbox_corners'] = [anno_2d['bbox'][0],anno_2d['bbox'][1],anno_2d['bbox'][0]+anno_2d['bbox'][2],anno_2d['bbox'][1]+anno_2d['bbox'][3]]
    instance_rec['filename'] = sd_rec['filename']
    instance_rec['instance_mask'] = anno_2d['segmentation']
    instance_rec['token'] = anno_2d['ann_token']
    return instance_rec

def main(args):
    """Generates 2D re-projections of the 3D bounding boxes present in the dataset."""

    print("Generating 2D instance mask and box of the nuScenes dataset ---> nuInsSeg")

    # Get tokens for all camera images.
    # sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if (s['sensor_modality'] == 'camera') and
    #                              s['is_key_frame']]
    annotations_2d = []
    if args.version == 'v1.0-mini':
        annos_root = os.path.join(args.dataroot, 'annotations')
        train_json = os.path.join(annos_root, 'refine2_nuscene_v1.0-mini_train_v1.0.json')
        val_json = os.path.join(annos_root, 'refine2_nuscene_v1.0-mini_val_v1.0.json')
    with open(train_json, 'r') as ft:
        train_data = json.load(ft)
        annotations_2d.extend(train_data['annotations'])
    with open(val_json, 'r') as fv:
        val_data = json.load(fv)
        annotations_2d.extend(val_data['annotations'])
    print('find {} 2d anntotations in {}'.format(len(annotations_2d), args.version))
    # For debugging purposes: Only produce the first n images.
    # if args.image_limit != -1:
    #     sample_data_camera_tokens = sample_data_camera_tokens[:args.image_limit]

    instance_masks = []
    for anno_2d in tqdm(annotations_2d):
        instance_record = get_2d_instance_mask(anno_2d)
        instance_masks.append(instance_record)

    # Save to a .json file.
    dest_path = os.path.join(args.dataroot, args.version)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with open(os.path.join(args.dataroot, args.version, args.filename), 'w') as fh:
        json.dump(instance_masks, fh, sort_keys=True, indent=4)

    print("Saved the 2D instance masks under {}".format(os.path.join(args.dataroot, args.version, args.filename)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export 2D annotations from reprojections to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/datasets/nuscenes', help="Path where nuScenes is saved.")
    parser.add_argument('--version', type=str, default='v1.0-mini', help='Dataset version.')
    parser.add_argument('--filename', type=str, default='nuinsseg.json', help='Output filename.')
    # parser.add_argument('--visibilities', type=str, default=['', '1', '2', '3', '4'],
    #                     help='Visibility bins, the higher the number the higher the visibility.', nargs='+')
    # parser.add_argument('--image_limit', type=int, default=-1, help='Number of images to process or -1 to process all.')
    args = parser.parse_args()
    args.dataroot = os.path.join(os.environ['HOME'], 'datasets/nuscenes')
    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    main(args)
