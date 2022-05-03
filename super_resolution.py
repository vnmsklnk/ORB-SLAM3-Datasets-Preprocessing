import argparse
import os
import cv2
import imageio as imageio
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("square_size")
    parser.add_argument("threshold")
    parser.add_argument("path_to_dataset")
    parser.add_argument("path_to_save")

    args = parser.parse_args()
    N = int(args.square_size)
    threshold = float(args.threshold)
    n1 = round((N - 1) / 2)
    n2 = round((N + 1) / 2)
    condition = N * N * threshold

    path_to_dataset = args.path_to_dataset
    path_to_depth = os.path.join(path_to_dataset, "depth")
    path_to_save = os.path.join(path_to_dataset, args.path_to_save)
    if not (os.path.isdir(path_to_save)):
        os.mkdir(path_to_save)

    images = os.listdir(path_to_depth)

    for image_raw in images:
        image = cv2.imread(os.path.join(path_to_depth, image_raw), -1)
        image_copy = image.copy()
        width, height = image.shape[:2][::-1]

        for i in range(n1, width - n1):
            for j in range(n1, height - n1):
                if image[j][i] == 0:
                    cropped = image_copy[j - n1 : j + n2, i - n1 : i + n2]
                    non_zero = np.count_nonzero(cropped)
                    if non_zero > condition:
                        image[j][i] = np.average(
                            cropped.ravel()[np.flatnonzero(cropped)]
                        )

        imageio.imwrite(os.path.join(path_to_save, image_raw), image)
