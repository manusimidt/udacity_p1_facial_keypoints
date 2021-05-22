import glob
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

key_pts_frame = pd.read_csv('data/training_frames_keypoints.csv')


def show_keypoints(image_nr: int):
    """Show image with keypoints"""
    image_name = key_pts_frame.iloc[image_nr, 0]
    key_pts = key_pts_frame.iloc[image_nr, 1:].as_matrix()
    key_pts = key_pts.astype('float').reshape(-1, 2)
    image = mpimg.imread(os.path.join('data/training/', image_name))
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
    plt.show()


if __name__ == '__main__':
    # show four random images with keyframes from the test dataset
    for x in range(4):
        show_keypoints(int(random.random() * 10))
