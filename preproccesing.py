import json
import multiprocessing
import os
from functools import partial
from multiprocessing import Pool
import argparse
import cv2
import numpy as np
import open3d
import configparser
from imageio import imwrite


def undistort(path, fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2):
    image_ = cv2.imread(path, -1)
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
    undistorted = cv2.undistort(image_, camera_matrix, np.array([k1, k2, p1, p2, k3, k4, k5, k6]))
    return undistorted


def xyz_to_uvz(point, fx, fy, cx, cy):
    x, y, z = point
    u = round(((fx * x) / z) + cx)
    v = round(((fy * y) / z) + cy)
    return u, v, z * 1000


def depth_images_preprocessing(depth_images_path, fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2,
                               width, height, width_color, height_color, transform_matrix, fx_c, fy_c, cx_c, cy_c):
    depth_images = os.listdir(depth_images_path)
    new_dir = os.path.join(os.path.dirname(depth_images_path), "depth_preprocessed")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    for image in depth_images:
        # Depth undistortion
        depth_undistorted = undistort(os.path.join(depth_images_path, image), fx, fy, cx, cy, k1, k2, k3, k4, k5, k6,
                                      p1, p2)

        # Point cloud creating
        picture = open3d.geometry.Image(depth_undistorted)
        intrinsic = open3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        pc = open3d.geometry.PointCloud()
        extrinsic = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        point_cloud = pc.create_from_depth_image(picture, intrinsic, extrinsic)

        # Point cloud transformation
        transformation_matrix = np.array(transform_matrix)
        point_cloud = point_cloud.transform(transformation_matrix)

        # Point cloud to depth image
        img = np.zeros((height_color, width_color))
        with Pool(multiprocessing.cpu_count()) as p:
            new_points = (p.map(partial(xyz_to_uvz, fx=fx_c, fy=fy_c, cx=cx_c, cy=cy_c), point_cloud.points))
            for u_, v_, z_ in new_points:
                try:
                    img[v_, u_] = z_
                except IndexError:
                    pass
            img = img.astype(np.uint16)
            imwrite(os.path.join(new_dir, image), img)


def color_images_preprocessing(color_images_path, fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2):
    color_images = os.listdir(color_images_path)
    new_dir = os.path.join(os.path.dirname(color_images_path), "color_preprocessed")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    # Color undistortion
    for image in color_images:
        color_undistorted = undistort(os.path.join(color_images_path, image), fx, fy, cx, cy, k1, k2, k3, k4, k5, k6,
                                      p1, p2)
        imwrite(os.path.join(new_dir, image), color_undistorted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('depth_images')
    parser.add_argument('color_images')
    parser.add_argument('config_path')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_path)
    settings = config['DEPTH']
    transform = json.loads(config['TRANSFORM']['matrix'])
    color_fx = float(config['COLOR']['fx'])
    color_fy = float(config['COLOR']['fx'])
    color_cx = float(config['COLOR']['fx'])
    color_cy = float(config['COLOR']['fx'])
    depth_images_preprocessing(args.depth_images, float(settings['fx']), float(settings['fy']), float(settings['cx']),
                               float(settings['cy']), float(settings['k1']), float(settings['k2']), float(settings['k3']), float(settings['k4']),
                               float(settings['k5']), float(settings['k6']), float(settings['p1']), float(settings['p2']),
                               int(settings['width']), int(settings['height']), int(config['COLOR']['width']),
                               int(config['COLOR']['height']), transform, color_fx, color_fy, color_cx, color_cy)
    settings = config['COLOR']
    color_images_preprocessing(args.color_images, color_fx, color_fy, color_cx, color_cy, float(settings['k1']),
                               float(settings['k2']), float(settings['k3']), float(settings['k4']), float(settings['k5']), float(settings['k6']),
                               float(settings['p1']), float(settings['p2']))
