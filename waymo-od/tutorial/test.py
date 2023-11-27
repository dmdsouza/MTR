import os
from typing import List, Dict, Tuple, Optional, Any
import sys
sys.path.insert(0, '/scratch1/dmdsouza/MTR/waymo-od/src')
import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import compressed_lidar_pb2
from waymo_open_dataset.utils import womd_lidar_utils

def _load_scenario_data(tfrecord_file: str) -> scenario_pb2.Scenario:
  """Load a scenario proto from a tfrecord dataset file."""
  dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
  data = next(iter(dataset))
  return scenario_pb2.Scenario.FromString(data.numpy())

WOMD_FILE = '/scratch1/dmdsouza/scenario/training/training.tfrecord-00000-of-01000'
womd_original_scenario = _load_scenario_data(WOMD_FILE)
print(womd_original_scenario.scenario_id)


LIDAR_DATA_FILE = '/scratch1/dmdsouza/lidar/training/4b60f9400a30ceaf.tfrecord'
womd_lidar_scenario = _load_scenario_data(LIDAR_DATA_FILE)
scenario_augmented = womd_lidar_utils.augment_womd_scenario_with_lidar_points(
    womd_original_scenario, womd_lidar_scenario)
print(len(scenario_augmented.compressed_frame_laser_data))

frame_points_xyz = {}  # map from frame indices to point clouds
frame_points_feature = {}
frame_i = 0

def _get_laser_calib(
    frame_lasers: compressed_lidar_pb2.CompressedFrameLaserData,
    laser_name: dataset_pb2.LaserName.Name):
  for laser_calib in frame_lasers.laser_calibrations:
    if laser_calib.name == laser_name:
      return laser_calib
  return None

# Extract point cloud xyz and features from each LiDAR and merge them for each
# laser frame in the scenario proto.
for frame_lasers in scenario_augmented.compressed_frame_laser_data:
  points_xyz_list = []
  points_feature_list = []
  frame_pose = np.reshape(np.array(
      scenario_augmented.compressed_frame_laser_data[frame_i].pose.transform),
      (4, 4))
  for laser in frame_lasers.lasers:
    if laser.name == dataset_pb2.LaserName.TOP:
      c = _get_laser_calib(frame_lasers, laser.name)
      (points_xyz, points_feature,
       points_xyz_return2,
       points_feature_return2) = womd_lidar_utils.extract_top_lidar_points(
           laser, frame_pose, c)
    else:
      c = _get_laser_calib(frame_lasers, laser.name)
      (points_xyz, points_feature,
       points_xyz_return2,
       points_feature_return2) = womd_lidar_utils.extract_side_lidar_points(
           laser, c)
    points_xyz_list.append(points_xyz.numpy())
    points_xyz_list.append(points_xyz_return2.numpy())
    points_feature_list.append(points_feature.numpy())
    points_feature_list.append(points_feature_return2.numpy())
  frame_points_xyz[frame_i] = np.concatenate(points_xyz_list, axis=0)
  frame_points_feature[frame_i] = np.concatenate(points_feature_list, axis=0)
  frame_i += 1


print(frame_points_xyz[0].shape)
print(frame_points_feature[0].shape)
