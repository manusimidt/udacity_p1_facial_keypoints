import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
from models import Net

# load in color image for face detection
image = cv2.imread('images/obamas.jpg')

net = Net()
net.load_state_dict(torch.load('saved_models/keypoints_model_1.pt'))

net.eval()

# switch red and blue color channels
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(image, 1.2, 2)

image_copy = np.copy(image)


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


# loop over the detected faces from your haar cascade
for (x, y, w, h) in faces:
    # Select the region of interest that is the face in the image
    roi = image_copy[y:y + h, x:x + w]

    # Convert the face region from RGB to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi_gray = cv2.normalize(roi_gray, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    roi_gray = cv2.resize(roi_gray, (224, 224))

    # Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    roi_gray_t = np.expand_dims(roi_gray, axis=0)
    roi_gray_t = np.expand_dims(roi_gray_t, axis=0)
    roi_gray_t = torch.from_numpy(roi_gray_t).float()
    predicted_points = net.forward(roi_gray_t)
    predicted_points = (predicted_points * 50.0 + 100)
    show_all_keypoints(roi, np.squeeze(predicted_points))
