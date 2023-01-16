import argparse
import cv2
import os

from pyk4a import PyK4APlayback, ImageFormat


def undistort(image, camera_matrix, dist_coeff):
    shape = image.shape[:2][::-1]
    undist_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeff, shape, 1, shape
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeff, None, undist_intrinsics, shape, cv2.CV_32FC1
    )
    undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST)
    return undistorted, undist_intrinsics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_mkv")
    parser.add_argument("path_to_save_output")
    args = parser.parse_args()

    playback = PyK4APlayback(args.path_to_mkv)
    playback.open()

    distortion = playback.calibration.get_distortion_coefficients(1)
    intrinsics = playback.calibration.get_camera_matrix(1)

    idx = 0
    color_path = os.path.join(args.path_to_save_output, "color")
    depth_path = os.path.join(args.path_to_save_output, "depth")
    os.mkdir(args.path_to_save_output)
    os.mkdir(color_path)
    os.mkdir(depth_path)
    while True:
        try:
            capture = playback.get_next_capture()
        except EOFError:
            break

        if (capture.color is None) or (capture.transformed_depth is None):
            continue

        color_image = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        depth_image = capture.transformed_depth

        color_undistorted, _ = undistort(color_image, intrinsics, distortion)
        depth_undistorted, new_intrinsics = undistort(
            depth_image, intrinsics, distortion
        )

        if idx == 0:
            with open(
                os.path.join(args.path_to_save_output, "new_color_intrinsics.txt"), "w"
            ) as file:
                file.write(str(new_intrinsics))

        color_timestamp = capture.color_timestamp_usec
        depth_timestamp = capture.depth_timestamp_usec

        color_filename = os.path.join(color_path, f'{color_timestamp:012d}.png')
        depth_filename = os.path.join(depth_path, f'{depth_timestamp:012d}.png')

        cv2.imwrite(color_filename, color_undistorted)
        cv2.imwrite(depth_filename, depth_undistorted)

        idx += 1

    playback.close()
