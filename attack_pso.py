import itertools

import numpy as np
import logging
import copy
import pyswarms as ps
import torch
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy.linalg import norm
import json

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import ctypes
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import PIXOR
from loss import CustomLoss

from utils_attack import *
from utils_nuscs import *
from utils import get_bev
from postprocess import compute_iou, convert_format, non_max_suppression
from utils import get_model_name, load_config

from export_kitti import *
from utils_attack import center_to_addpts

def get_continue_multi_frame(target_res):
    t1 = target_res[0]
    t2 = target_res[1]
    t3 = target_res[2]
    t4 = target_res[3]
    t5 = target_res[4]
    combinations = list(itertools.product(t1, t2, t3, t4, t5))
    return combinations

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def calculate_D_max(H1, H2):
    distances = np.sum(np.square(H1[:, :2] - H2[:, :2]), axis=1)
    
    D_max = np.max(distances)
    if D_max == 0:
        D_max = 1e-6
    # D_max = 16
    return D_max

def compute_similarity_loss(det_res, target_res):
    '''
    Args:
        det_res: randomly generated detection results
        target_res: inverse attack target results
    Returns: similarity loss
    '''
    # 1. ade
    D_max = calculate_D_max(det_res, target_res)
    ade = np.sum(np.square(det_res[:, :2] - target_res[:, :2])) / len(target_res)
    ade_norm = ade / D_max
    position_similarity = 1 - ade_norm
    # position_distances = np.mean([euclidean(det_res[i, :2], target_res[i, :2]) for i in range(len(det_res))])

    # 2. cos
    heading_similarities = np.mean([cosine_similarity(det_res[i, 2:], target_res[i, 2:]) for i in range(len(det_res))])

    # 3. DTW
    dtw_distance, _ = fastdtw(det_res[:, :2], target_res[:, :2], dist=euclidean)

    alpha = 0.4  
    beta = 0.2   
    gamma = 0.4 

    similarity_score = alpha * position_similarity + beta * heading_similarities + gamma * (1 - dtw_distance)
    return 1 - similarity_score

def objective(x_birds, net, config, device, sample_tokens, scene_name, target_res, nusc):
    data_dir = '/dataset/nuScenes/mini_attack'
    dir_names = os.listdir(data_dir)
    dir_names = [s for s in dir_names if s.startswith(scene_name)]
    dir_name = dir_names[0]
    dataset_dir = os.path.join(data_dir, dir_name)
    config_path = os.path.join(dataset_dir, 'config.yaml')
    cfg = cfg_from_yaml_file(config_path)
    cfg = cfg.DET

    frame_ids = sample_tokens
    lidar_paths = [os.path.join(dataset_dir, 'velodyne', f_id + '.bin') for f_id in frame_ids]

    N = x_birds.shape[0]
    loss = np.zeros(x_birds.shape[0])

    for n in range(N):
        # used for vague optimization
        added_points = x_birds[n, :].reshape((cfg.N_add, 3))  # n
        added_points = center_to_addpts(1, cfg.N_add, cfg.Npts_cls, 0.2, added_points)

        det_res = []

        for f_idx in range(len(frame_ids)):
            frame_id = frame_ids[f_idx]

            lidar_path = lidar_paths[f_idx]
            label_list = []
            w, h, l, y, z, x, yaw = get_gt3Dboxes(dataset_dir, frame_id)[0]

            ### get ground truth ###
            bev_corners = np.zeros((4, 2), dtype=np.float32)
            # rear left
            bev_corners[0, 0] = x - l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw)
            bev_corners[0, 1] = y - l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)
            # rear right
            bev_corners[1, 0] = x - l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw)
            bev_corners[1, 1] = y - l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)
            # front right
            bev_corners[2, 0] = x + l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw)
            bev_corners[2, 1] = y + l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)
            # front left
            bev_corners[3, 0] = x + l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw)
            bev_corners[3, 1] = y + l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)
            # reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]
            geom = config['geometry']['input_shape']
            # gt_reg = np.array([[1.0,x,y,w,l,yaw]])
            # print(gt_reg)
            label_list = bev_corners[np.newaxis, :]

            # gt3Dboxes = np.array([[w, h, l, y, z, x, yaw]])
            kitti_token = '%s_%s' % (dir_name, frame_id)
            gt_box_lidar = get_gt_box(token=kitti_token, root=data_dir)  # have some problem
            x1 = gt_box_lidar.center[0]
            y1 = gt_box_lidar.center[1]
            z1 = gt_box_lidar.center[2]
            w1 = gt_box_lidar.wlh[0]
            l1 = gt_box_lidar.wlh[1]
            h1 = gt_box_lidar.wlh[2]
            yaw1 = gt_box_lidar.orientation.yaw_pitch_roll[0]
            gt3Dboxes = np.array([[w1, h1, l1, y1, z1, x1, yaw1]])  # Lidar frame system

            score = attack_obj_nuscs(added_points, net, 0, config, geom, lidar_path, label_list, gt3Dboxes, device, cfg.N_add * cfg.Npts_cls)

            s, x, y, w, l, yaw = score
            res = [float(s), float(x), float(y), z, float(l), float(w), h, float(yaw)]
            parsed_line = process_res_to_lidar_kitti(res, kitti_token, data_dir)
            x = parsed_line['x_lidar']
            y = parsed_line['y_lidar']
            wlh = (parsed_line['w'], parsed_line['l'], 0)
            yaw_lidar = parsed_line['yaw_lidar']
            quat_box = Quaternion(axis=(0, 0, 1), angle=yaw_lidar)
            box = Box([x, y, 0], wlh, quat_box, name='car')
            # 4: Transform to nuScenes LIDAR coord system.
            kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
            box.rotate(kitti_to_nu_lidar)

            box_global = lidar_nusc_box_to_global(nusc, [box], frame_id)[0]
            target_x_global, target_y_global, target_heading_global = box_global.center[0], box_global.center[1], \
                box_global.orientation.yaw_pitch_roll[0]

            det_res.append([target_x_global, target_y_global, target_heading_global])

        _loss = compute_similarity_loss(np.array(det_res), target_res)
        loss[n] += _loss
    return loss
  
def get_bounds_cls(N_add, Npts_cls):
    lower_bound = np.array([-2.1, -2.1, -0.1, 0.4])
    upper_bound = np.array([2.1, 2.1, 0.9, 0.7])
    lower_bound = np.tile(lower_bound, (N_add * Npts_cls, 1)).reshape(N_add*Npts_cls*4)
    upper_bound = np.tile(upper_bound, (N_add * Npts_cls, 1)).reshape(N_add*Npts_cls*4)
    center = (lower_bound + upper_bound) / 2
    return lower_bound, upper_bound, center

def get_bounds_center(N_add):
    lower_bound = np.array([-2., -2., 0.])
    upper_bound = np.array([2., 2., 0.8])
    lower_bound = np.tile(lower_bound, (N_add, 1)).reshape(N_add*3)
    upper_bound = np.tile(upper_bound, (N_add, 1)).reshape(N_add*3)
    center = (lower_bound + upper_bound) / 2
    return lower_bound, upper_bound, center

class BaseAttacker():
    def __init__(self, obs_length, pred_length, net, config, device):
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.net = net
        self.config = config
        self.device = device

class PSOAttacker(BaseAttacker):
    def __init__(self, obs_length, pred_length, net, config, device, nusc, n_particles=10, iter_num=100, c1=2, c2=2, w=1.0, bound=np.array([2, 2, 0.8]), physical_bounds={}):
        super().__init__(obs_length, pred_length, net, config, device)
        self.iter_num = iter_num
        self.bound = bound
        self.options = {'c1': c1, 'c2': c2, 'w': w}
        self.n_particles = n_particles
        self.nusc = nusc

    def run(self, sample_tokens, scene_name, target_res):
        # lower_bound, upper_bound, center = get_bounds_cls(3, 4)
        target_res = target_res # todo: read target_res from file
        lower_bound, upper_bound, center = get_bounds_center(3)
        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=3 * 3, options=self.options,
                                            bounds=(lower_bound, upper_bound),
                                            center=center)

        best_loss, best_perturb = optimizer.optimize(objective, iters=self.iter_num, net=self.net, config=self.config,
                                                     device=self.device, sample_tokens=sample_tokens,
                                                     scene_name=scene_name, target_res=target_res, nusc=self.nusc)
        return best_loss, best_perturb


if __name__ == '__main__':
    scene_name = 'scene-0043' 
    multi_frame_target_path = './inverse_res_multi_frame.json'
    with open(multi_frame_target_path, 'r') as f:
        multi_frame_target = json.load(f)
    f.close()

    target_r = []
    for i in range(1,6):
        k = scene_name + '_{}'.format(str(i))
        target_res = multi_frame_target[k]
        pos = target_res['pos'][:1] 
        heading = target_res['heading'][:1]
        res = np.concatenate((pos, heading), axis=1)
        target_r.append(res)
    target_results = get_continue_multi_frame(target_r)

    # get attack frame ids
    sample_tokens = []
    with open(os.path.join('./target_sample_tokens.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample_token = line.strip()
            sample_tokens.append(sample_token)
    f.close()

    sample_tokens = sample_tokens[12 - 4: 12 + 1] 

    data_dir = './nuScenes/mini_attack'
    # init device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # init config
    config, _, _, _ = load_config('nusce_kitti')

    # init model
    net, loss_fn = build_model(config, device, train=False)
    net.load_state_dict(torch.load(get_model_name(config, epoch=38), map_location=device))
    net.set_decode(True)
    net.eval()

    nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuScenes/trainval', verbose=True)

    loss = []
    perturb = []
    for i in range(len(target_results)):
        res = np.zeros((5, 3))
        temp = target_results[i]
        for j in range(5):
            res[j] = temp[j]
        print('Step :', i)
        # pso for the best pos(x,y,z)
        attacker = PSOAttacker(4, 6, net, config, device, nusc, n_particles=10)
        best_loss, best_perturb =attacker.run(sample_tokens, scene_name, res)
        loss.append(best_loss)
        perturb.append(best_perturb.tolist())

    pso_res = {
        scene_name:{
            'loss': loss,
            'perturb': perturb,
        }
    }

    current_path = '/dataset/nuScenes/mini_attack/inverse/'
    with open(os.path.join(current_path, 'pso_res.json'), 'w') as f:
        json.dump(pso_res, f)




