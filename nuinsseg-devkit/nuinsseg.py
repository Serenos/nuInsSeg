from nuscenes.nuscenes import NuScenes
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from nuscenes.utils.color_map import get_colormap
from pycocotools import mask as cocomask
import copy
from pyquaternion import Quaternion

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points, points_in_box
    from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
except:
    print("nuScenes devkit not Found!")

# using cam and lidar data
CAM_SENSOR = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
POINT_SENSOR = 'LIDAR_TOP'

class NuInsSeg(NuScenes):
    '''
    '''
    def __init__(self, version: str = 'v1.0-mini', dataroot: str = '/data/sets/nuscenes', verbose: bool = True, map_resolution: float = 0.1):
        super().__init__(version, dataroot, verbose, map_resolution)
        # If available, also load the instance mask table created by export_2d_annotations_as_json().
        if os.path.exists(os.path.join(self.table_root, 'nuinsseg.json')):
            new_table = 'nuinsseg'
            self.table_names.append(new_table)
            self.nuinsseg = self.__load_table__(new_table)
            print('Loading nuInsSeg dataset from nuinsseg.json')
            print('{} instance masks'.format(len(self.nuinsseg)))
            self._token2ind[new_table] = dict()
            for ind, member in enumerate(getattr(self, new_table)):
                self._token2ind[new_table][member['token'], member['sample_data_token']] = ind
            self.color_map = get_colormap()

    def get_nuinsseg(self, ann_token, sd_token):
        index = self._token2ind['nuinsseg'][ann_token, sd_token]
        return self.nuinsseg[index]

    def render_2d_annotation(self, sample, sample_annotation_token) -> None:
        sd_tokens = [sample['data'][cam] for cam in CAM_SENSOR]
        instance_recs = []
        for sd_token in sd_tokens:
            if (sample_annotation_token, sd_token) in list(self._token2ind['nuinsseg'].keys()):
                instance_recs.append(self.get_nuinsseg(sample_annotation_token, sd_token))
        # Get image data.
        print('found {} annotations in different cams'.format(len(instance_recs)))
        for i, instance_rec in enumerate(instance_recs):
            im_path = os.path.join(self.dataroot, instance_rec['filename'])
            im = Image.open(im_path)

            im = im.convert('RGBA')
            draw = ImageDraw.Draw(im, 'RGBA')

            instance_rec['instance_mask'] = [poly for poly in instance_rec['instance_mask'] if len(poly)!=0]
            if instance_rec['instance_mask'] is not None:
                if len(instance_rec['instance_mask']) == 0: #filter empty
                    continue
                elif isinstance(instance_rec['instance_mask'][0], dict): #rle format
                    mask = cocomask.decode(instance_rec['instance_mask'])
                else: #polygon format
                    Rs = cocomask.frPyObjects(instance_rec['instance_mask'], 900, 1600)
                    mask = cocomask.decode(Rs)
                category_name = instance_rec['category_name']
                color = self.color_map[category_name]
                bbox = instance_rec['bbox_corners']
                # Draw mask, rectangle and text.
                if len(mask.shape)==3:
                    mask = mask[:,:,0]
                draw.bitmap((0, 0), Image.fromarray(mask * 128), fill=tuple(color + (128,)))
                draw.rectangle(bbox, outline=color, width=1)
                # if with_category:
                #     draw.text((bbox[0], bbox[1]), name, font=font)

                # Plot the image.
                (width, height) = im.size
                pix_to_inch = 100 / 1 #render_scale
                figsize = (height / pix_to_inch, width / pix_to_inch)
                plt.figure(figsize=figsize)
                plt.axis('off')
                plt.imshow(im)

    def render_2d_image(self, sd_token) -> None:
        sd = self.get('sample_data', sd_token)
        s = self.get('sample', sd['sample_token'])

        anns = []
        for ann_token in s['anns']:
            if (ann_token, sd_token) in list(self._token2ind['nuinsseg'].keys()):
                ann = self.get_nuinsseg(ann_token, sd_token)
                anns.append(ann)
        print('found {} annotations'.format(len(anns)))
        if len(anns) == 0:
            return 
        im_path = os.path.join(self.dataroot, anns[0]['filename'])
        im = Image.open(im_path)
        im = im.convert('RGBA')
        draw = ImageDraw.Draw(im, 'RGBA')
        for i, instance_rec in enumerate(anns):
            instance_rec['instance_mask'] = [poly for poly in instance_rec['instance_mask'] if len(poly)!=0]
            if instance_rec['instance_mask'] is not None:
                if len(instance_rec['instance_mask']) == 0: #filter empty
                    continue
                elif isinstance(instance_rec['instance_mask'][0], dict): #rle format
                    mask = cocomask.decode(instance_rec['instance_mask'])
                else: #polygon format
                    Rs = cocomask.frPyObjects(instance_rec['instance_mask'], 900, 1600)
                    mask = cocomask.decode(Rs)
                category_name = instance_rec['category_name']
                color = self.color_map[category_name]
                bbox = instance_rec['bbox_corners']
                # Draw mask, rectangle and text.
                if len(mask.shape)==3:
                    mask = mask[:,:,0]
                draw.bitmap((0, 0), Image.fromarray(mask * 128), fill=tuple(color + (128,)))
                draw.rectangle(bbox, outline=color, width=1)

        (width, height) = im.size
        pix_to_inch = 100 / 1 #render_scale
        figsize = (height / pix_to_inch, width / pix_to_inch)
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(im)     

    def list_categories_nuseg(self) -> None:
        """ Print categories, counts and stats. These stats only cover the split specified in nuseg.version. """
        print('NuInsSeg: Category stats for split %s:' % self.version)
        # Add all annotations.
        categories = dict()
        for record in self.nuinsseg:
            if record['category_name'] not in categories:
                categories[record['category_name']] = []
            categories[record['category_name']].append(record['num_lidar_pts'])

        # Print stats.
        for name, stats in sorted(categories.items()):
            stats = np.array(stats)
            print('{:35} n={:10}'.format(name[:35], stats.shape[0]))

    def map_point_to_img(self, nusc, pc, pointsensor, cam):
        img = Image.open(os.path.join(nusc.dataroot, cam['filename']))

        # projection
        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        pose_record = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
        pc.translate(np.array(pose_record['translation']))

        pose_record = nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(pose_record['translation']))
        pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix.T)

        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        depths = pc.points[2, :]
        coloring = depths

        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
        points = view_points(pc.points[:3, :], camera_intrinsic, normalize=True)

        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 1)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
        points = points[:, mask].astype(np.int16)
        coloring = coloring[mask]

        return points, coloring, img, mask

    def filter_with_2dbox(self, points, h, w, depth, bbox=None):
        mask = np.ones(points.shape[1], dtype=bool)
        if bbox:
            mask = np.logical_and(mask, depth > 1)
            mask = np.logical_and(mask, points[0, :] > max(bbox[0] + 1, 1))
            mask = np.logical_and(mask, points[0, :] < min(bbox[0] + bbox[2] - 1, w - 1))
            mask = np.logical_and(mask, points[1, :] > max(bbox[1] + 1, 1))
            mask = np.logical_and(mask, points[1, :] < min(bbox[1] + bbox[3] - 1, h - 1))
        else:
            mask = np.logical_and(mask, depth > 1)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < w - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < h - 1)
        points = points[:2, mask]
        depth = depth[mask]
        return points, depth, mask

    def render_pc_per_instance(self, sample_record, multisweeps=0, remove_overlap=0, cams=None) -> None:

        point_data_token = sample_record['data'][POINT_SENSOR]
        point_data = self.get('sample_data', point_data_token)
        lidar_path = point_data['filename']
        full_lidar_path = os.path.join(self.dataroot, lidar_path)
        if multisweeps:
            pc, times = LidarPointCloud.from_file_multisweep(self, sample_record, POINT_SENSOR, POINT_SENSOR)
        else:
            pc = LidarPointCloud.from_file(full_lidar_path)
        
        if cams == None:
            cams = CAM_SENSOR
        for cam in cams:
            camera_token = sample_record['data'][cam]
            cam_data = self.get('sample_data', camera_token)
            pc_temp = copy.deepcopy(pc)
            point2d, coloring, img, mask = self.map_point_to_img(self, pc_temp, point_data, cam_data)
            if remove_overlap:       
                #point3d, lidarseg = pc.points[:, mask], lidarseg_temp[mask]
                point3d = pc.points[:, mask]
                depth_map = np.zeros((img.size[1], img.size[0]))
                loc2index = np.zeros((img.size[1], img.size[0]), dtype=int)
                point2d = point2d[:2, :].astype(int)
                depth_map[point2d[1, :], point2d[0, :]] = coloring
                loc2index[point2d[1, :], point2d[0, :]] = [i for i in range(point2d.shape[1])]

                refine_depth_map = copy.deepcopy(depth_map)
                refine_depth_map = remove_overlap(depth_img=refine_depth_map)

                mask = np.ones(point2d.shape[1])
                temp = np.logical_and(depth_map > 0, refine_depth_map == 0)
                fliter_loc = temp.nonzero()
                points_index = loc2index[fliter_loc]
                mask[points_index] = 0
                mask = mask.astype(np.bool8)
                #pc2d, pc3d, depth, lidarseg = point2d[:, mask], point3d[:, mask], coloring[mask], lidarseg[mask]
                pc2d, pc3d, depth = point2d[:, mask], point3d[:, mask], coloring[mask]
            else:
                point3d = pc.points[:, mask]
                pc2d, pc3d, depth = point2d, point3d, coloring
            
            sd_token = sample_record['data'][cam]
            sd = self.get('sample_data', sd_token)
            s = self.get('sample', sd['sample_token'])

            im_path = os.path.join(self.dataroot, sd['filename'])
            im = Image.open(im_path)
            im = im.convert('RGBA')
            draw = ImageDraw.Draw(im, 'RGBA')
            (width, height) = im.size
            pix_to_inch = 100 / 1 #render_scale
            figsize = (height / pix_to_inch, width / pix_to_inch)
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            anns = []
            for ann_token in s['anns']:
                if (ann_token, sd_token) in list(self._token2ind['nuinsseg'].keys()):
                    ann = self.get_nuinsseg(ann_token, sd_token)
                    anns.append(ann)
            print('found {} annotations'.format(len(anns)))

            ann_tokens = [ann['token'] for ann in anns]
            _, box_lidar_frame, _ = self.get_sample_data(sample_record['data'][POINT_SENSOR], selected_anntokens=ann_tokens)
            for i, box in enumerate(box_lidar_frame):
                min_x, min_y, max_x, max_y = anns[i]['bbox_corners']
                box2d_xywh = [min_x, min_y, max_x-min_x, max_y-min_y]
                logits = points_in_box(box, pc3d[:3, :])
                pc2d_inbox = copy.deepcopy(pc2d)
                pc2d_outbox = copy.deepcopy(pc2d)
                pc2d_inbox = pc2d_inbox[:, logits]
                pc2d_outbox = pc2d_outbox[:, ~logits]

                depth_inbox = copy.deepcopy(depth)
                depth_outbox = copy.deepcopy(depth)
                depth_inbox = depth_inbox[logits]
                depth_outbox = depth_outbox[~logits]
                
                pc3d_inbox = copy.deepcopy(pc3d)
                pc3d_inbox = pc3d_inbox[:, logits]
                pc3d_outbox = copy.deepcopy(pc3d)
                pc3d_outbox = pc3d_outbox[:, ~logits]

                pc2d_inbox, depth_inbox, maskin = self.filter_with_2dbox(pc2d_inbox, img.size[1], img.size[0], depth_inbox)
                pc2d_outbox, depth_outbox, maskout = self.filter_with_2dbox(pc2d_outbox, img.size[1], img.size[0], depth_outbox, box2d_xywh)
                pc3d_inbox = pc3d_inbox[:, maskin]
                pc3d_outbox = pc3d_outbox[:, maskout]
                pc3d_inbox = np.concatenate((pc3d_inbox, pc2d_inbox), axis=0)
                pc3d_outbox = np.concatenate((pc3d_outbox, pc2d_outbox), axis=0)

                ax.scatter(np.array(pc3d_inbox)[4, :], np.array(pc3d_inbox)[5, :], c='r', s=10)
                ax.scatter(np.array(pc3d_outbox)[4, :], np.array(pc3d_outbox)[5, :], c='b', s=5)
                rect = plt.Rectangle((box2d_xywh[0], box2d_xywh[1]), box2d_xywh[2], box2d_xywh[3], fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)

            ax.axis('off')
            ax.imshow(im)
            pass

if __name__ == '__main__':
    dataroot = os.path.join(os.environ['HOME'], 'datasets/nuscenes')
    nuseg = NuInsSeg(version='v1.0-mini', dataroot=dataroot, verbose=True)

    nuseg.list_categories_nuseg()
    #test instance mask render
    my_scene = nuseg.scene[2]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nuseg.get('sample', first_sample_token)
    my_annotation_token = my_sample['anns'][14]
    my_annotation_metadata =  nuseg.get('sample_annotation', my_annotation_token)
    nuseg.render_2d_annotation(my_sample, my_annotation_token)

    #nuseg.render_pc_per_instance(my_sample)