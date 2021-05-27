import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2

# load in training data
key_pts_frame = pd.read_csv('data/training_frames_keypoints.csv')

# print out some stats about the data
print('Number of images: ', key_pts_frame.shape[0])

n = 120
image_name = key_pts_frame.iloc[n, 0]
image = mpimg.imread(os.path.join('data/training/', image_name))
key_pts = key_pts_frame.iloc[n, 1:].as_matrix()
key_pts = key_pts.astype('float').reshape(-1, 2)

print('Image name: ', image_name)

plt.figure(figsize=(5, 5))

# copy of the face image for overlay
image_copy = np.copy(image)

# top-left location for sunglasses to go
# 17 = edge of left eyebrow
x = int(key_pts[17, 0])
y = int(key_pts[17, 1])

# height and width of sunglasses
# h = length of nose
h = int(abs(key_pts[27, 1] - key_pts[34, 1]))
# w = left to right eyebrow edges
w = int(abs(key_pts[17, 0] - key_pts[26, 0]))

# read in sunglasses
sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)
# resize sunglasses
new_sunglasses = cv2.resize(sunglasses, (w, h), interpolation=cv2.INTER_CUBIC)

# get region of interest on the face to change
roi_color = image_copy[y:y + h, x:x + w]

# find all non-transparent pts
ind = np.argwhere(new_sunglasses[:, :, 3] > 0)

# for each non-transparent point, replace the original image pixel with that of the new_sunglasses
for i in range(3):
    roi_color[ind[:, 0], ind[:, 1], i] = new_sunglasses[ind[:, 0], ind[:, 1], i]
# set the area of the image to the changed region with sunglasses
image_copy[y:y + h, x:x + w] = roi_color

# display the result!
plt.imshow(image_copy)
plt.show()
