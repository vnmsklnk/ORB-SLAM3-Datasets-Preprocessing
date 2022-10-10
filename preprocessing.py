import json
import os
import argparse

import cv2
import numpy as np
import open3d
import configparser
import imageio
from tqdm import tqdm


def undistort(path, camera_matrix, dist_coeff):
    """
    Undistorts image given
    :param path: path to the image for undistorting
    :param camera_matrix: camera's intrinsic parameters
    :param dist_coeff: list of k1, k2, p1, p2, k3, k4, k5, k6 coefficients
    :return: undistorted image and new camera's intrinsics
    """
    image = cv2.imread(path, -1)
    shape = image.shape[:2][::-1]
    undist_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeff, shape, 1, shape
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeff, None, undist_intrinsics, shape, cv2.CV_32FC1
    )
    undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST)
    return undistorted, undist_intrinsics


def reprojection(
    depth_undistorted,
    undist_intrinsics,
    intrinsic,
    transform_matrix,
    color_matrix,
    shape,
):
    """
    Reprojects the points of the depth image onto the coordinates of the color image
    :param depth_undistorted: undistorted depth image
    :param undist_intrinsics: camera's intrinsics after undistort
    :param intrinsic: open3d.camera.PinholeCameraIntrinsic instance
    :param transform_matrix: transformation matrix from one coordinate system to another
    :param color_matrix: color camera's intrinsics
    :param shape: shape of color images
    :return: new depth image in the color image coordinates
    """
    picture = open3d.geometry.Image(depth_undistorted)
    intrinsic.intrinsic_matrix = undist_intrinsics
    pc = open3d.geometry.PointCloud()
    point_cloud = pc.create_from_depth_image(picture, intrinsic)

    point_cloud = point_cloud.transform(transform_matrix)
    points = np.concatenate(
        (np.asarray(point_cloud.points), np.full((len(point_cloud.points), 1), 1)),
        axis=1,
    )

    result = color_matrix @ points.T
    result[:2] /= result[2]
    result[2] *= 1000
    result = np.round(result).astype(int)
    w, h = shape
    result = result[
        :, (result[0] >= 0) & (result[0] < w) & (result[1] >= 0) & (result[1] < h)
    ]
    img = np.zeros((h, w)).astype(np.uint16)

    def f(point):
        u, v, z, _ = point
        img[v, u] = z

    np.apply_along_axis(f, 0, result)
    return img


def depth_images_preprocessing(
    dataset_path,
    depth_matrix,
    dist_coeff,
    shape_depth,
    shape_color,
    transform_matrix,
    color_matrix,
    folder_name,
):
    """
    Preprocesses depth images. Preprocessing includes undistorting and reprojection.
    :param dataset_path: path to dataset with images
    :param depth_matrix: depth camera's intrinsics
    :param dist_coeff: list of k1, k2, p1, p2, k3, k4, k5, k6 coefficients
    :param shape_depth: shape of depth images
    :param shape_color: shape of color images
    :param transform_matrix: transformation matrix from one coordinate system to another
    :param color_matrix: color camera's intrinsics
    :param folder_name: folder name for saving results of preprocessing
    """
    depth_images_folder = os.path.join(dataset_path, "depth")
    depth_images = os.listdir(depth_images_folder)
    new_dir = os.path.join(dataset_path, folder_name)
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    ext_color_matrix = np.eye(4)
    ext_color_matrix[:3, :3] = color_matrix
    intrinsic = open3d.camera.PinholeCameraIntrinsic()
    intrinsic.width, intrinsic.height = shape_depth
    for image in tqdm(depth_images):
        depth_undistorted, undist_intrinsics = undistort(
            os.path.join(depth_images_folder, image), np.array(depth_matrix), dist_coeff
        )

        img = reprojection(
            depth_undistorted,
            undist_intrinsics,
            intrinsic,
            transform_matrix,
            ext_color_matrix,
            shape_color,
        )
        cv2.imwrite(os.path.join(new_dir, image), img)


def color_images_preprocessing(dataset_path, camera_matrix, dist_coeff, folder_name):
    """
    Undistorts color images
    :param dataset_path: path to dataset with images
    :param camera_matrix: color camera's intrinsics
    :param dist_coeff: list of k1, k2, p1, p2, k3, k4, k5, k6 coefficients
    :param folder_name: folder name for saving results of preprocessing
    """
    color_images_folder = os.path.join(dataset_path, "color")
    color_images = os.listdir(color_images_folder)
    new_dir = os.path.join(dataset_path, folder_name)
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    for image in tqdm(color_images):
        color_undistorted, _ = undistort(
            os.path.join(color_images_folder, image), camera_matrix, dist_coeff
        )
        cv2.imwrite(os.path.join(new_dir, image), color_undistorted)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_folder")
    parser.add_argument("path_to_save_color")
    parser.add_argument("path_to_save_depth")
    parser.add_argument("config_path")

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_path)

    def get_intrinsics(settings):
        dist_coeff = [
            float(settings["k1"]),
            float(settings["k2"]),
            float(settings["p1"]),
            float(settings["p2"]),
            float(settings["k3"]),
            float(settings["k4"]),
            float(settings["k5"]),
            float(settings["k6"]),
        ]
        camera_matrix = [
            [float(settings["fx"]), 0, float(settings["cx"])],
            [0, float(settings["fy"]), float(settings["cy"])],
            [0, 0, 1],
        ]
        return np.asarray(dist_coeff), camera_matrix

    depth_dist_coeff, depth_camera_matrix = get_intrinsics(config["DEPTH"])
    color_dist_coeff, color_camera_matrix = get_intrinsics(config["COLOR"])
    depth_shape = int(config["DEPTH"]["width"]), int(config["DEPTH"]["height"])
    color_shape = int(config["COLOR"]["width"]), int(config["COLOR"]["height"])
    transform = np.array(json.loads(config["TRANSFORM"]["matrix"]))

    depth_images_preprocessing(
        args.path_to_folder,
        depth_camera_matrix,
        depth_dist_coeff,
        depth_shape,
        color_shape,
        transform,
        np.array(color_camera_matrix),
        args.path_to_save_depth,
    )

    color_images_preprocessing(
        args.path_to_folder,
        np.array(color_camera_matrix),
        color_dist_coeff,
        args.path_to_save_color,
    )
