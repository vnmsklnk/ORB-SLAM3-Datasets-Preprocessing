import json
import os
import argparse
import cv2
import numpy as np
import open3d
import configparser
from imageio import imwrite


def undistort(path, camera_matrix, k_vector, p_vector):
    image = cv2.imread(path, -1)
    k1, k2, k3, k4, k5, k6 = k_vector
    p1, p2 = p_vector
    undistorted = cv2.undistort(
        image, camera_matrix, np.array([k1, k2, p1, p2, k3, k4, k5, k6])
    )
    return undistorted


def depth_images_preprocessing(
    depth_images_path,
    depth_matrix,
    k_vector,
    p_vector,
    width,
    height,
    width_color,
    height_color,
    transform_matrix,
    color_matrix,
):
    depth_images = os.listdir(depth_images_path)
    new_dir = os.path.join(os.path.dirname(depth_images_path), "depth_preprocessed")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    color_matrix = np.array(color_matrix)
    color_matrix = np.concatenate((color_matrix, np.zeros((3, 1))), axis=1)
    color_matrix = np.concatenate((color_matrix, np.array([[0, 0, 0, 1]])), axis=0)
    intrinsic = open3d.camera.PinholeCameraIntrinsic()
    intrinsic.width = width
    intrinsic.height = height
    intrinsic.intrinsic_matrix = depth_matrix
    pc = open3d.geometry.PointCloud()
    transformation_matrix = np.array(transform_matrix)
    for image in depth_images:
        # Depth undistortion
        depth_undistorted = undistort(
            os.path.join(depth_images_path, image),
            np.array(depth_matrix),
            k_vector,
            p_vector,
        )

        # Point cloud creating
        picture = open3d.geometry.Image(depth_undistorted)
        point_cloud = pc.create_from_depth_image(picture, intrinsic)

        # Point cloud transformation
        point_cloud = point_cloud.transform(transformation_matrix)

        # Point cloud to depth image
        points = np.concatenate(
            (np.asarray(point_cloud.points), np.full((len(point_cloud.points), 1), 1)),
            axis=1,
        )
        result = np.matmul(color_matrix, np.matrix.transpose(points))
        img = np.zeros((height_color, width_color)).astype(np.uint16)
        for point in np.matrix.transpose(result):
            uz, vz, z, _ = point
            u = round(uz / z)
            v = round(vz / z)
            try:
                img[v, u] = z * 1000
            except IndexError:
                pass
        imwrite(os.path.join(new_dir, image), img)


def color_images_preprocessing(color_images_path, camera_matrix, k_vector, p_vector):
    color_images = os.listdir(color_images_path)
    new_dir = os.path.join(os.path.dirname(color_images_path), "color_preprocessed")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    # Color undistortion
    for image in color_images:
        color_undistorted = undistort(
            os.path.join(color_images_path, image),
            np.array(camera_matrix),
            k_vector,
            p_vector,
        )
        imwrite(os.path.join(new_dir, image), color_undistorted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("depth_images")
    parser.add_argument("color_images")
    parser.add_argument("config_path")

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_path)

    def get_intrinsics(settings):
        k_vector = [
            float(settings["k1"]),
            float(settings["k2"]),
            float(settings["k3"]),
            float(settings["k4"]),
            float(settings["k5"]),
            float(settings["k6"]),
        ]
        p_vector = [float(settings["p1"]), float(settings["p2"])]
        camera_matrix = [
            [float(settings["fx"]), 0, float(settings["cx"])],
            [0, float(settings["fy"]), float(settings["cy"])],
            [0, 0, 1],
        ]
        return k_vector, p_vector, camera_matrix

    k_depth_vector, p_depth_vector, depth_camera_matrix = get_intrinsics(
        config["DEPTH"]
    )
    k_color_vector, p_color_vector, color_camera_matrix = get_intrinsics(
        config["COLOR"]
    )
    depth_width, depth_height = int(config["DEPTH"]["width"]), int(
        config["DEPTH"]["height"]
    )
    color_width, color_height = int(config["COLOR"]["width"]), int(
        config["COLOR"]["height"]
    )
    transform = json.loads(config["TRANSFORM"]["matrix"])

    depth_images_preprocessing(
        args.depth_images,
        depth_camera_matrix,
        k_depth_vector,
        p_depth_vector,
        depth_width,
        depth_height,
        color_width,
        color_height,
        transform,
        color_camera_matrix,
    )

    color_images_preprocessing(
        args.color_images, color_camera_matrix, k_color_vector, p_color_vector
    )
