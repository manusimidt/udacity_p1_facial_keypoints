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

net.eval()
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
    roi_gray_t = np.expand_dims(roi_gray, 0)
    roi_gray_t = torch.from_numpy(np.expand_dims(roi_gray_t, 0)).float()

    predicted_points = net.forward(roi_gray_t).data.view(68, -1).numpy()
    # undo normalization of keypoints
    predicted_points = (predicted_points * 100.0 + 90)

    fig = plt.figure(figsize=(5, 5))
    plot = fig.add_subplot(1, 1, 1)
    plot.imshow(np.squeeze(roi_gray), cmap='gray')
    plot.scatter(predicted_points[:, 0], predicted_points[:, 1], s=20, marker='.', c='m')
    plt.show()
