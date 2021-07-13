import os
import sys
import cv2
import numpy as np

from pyagender.wide_resnet import WideResNet

faceCascade_model = "./pretrained_models/haarcascade_frontalface_default.xml"
agModel = "./pretrained_models/weights.28-3.73.hdf5"
faceCascade = cv2.CascadeClassifier(faceCascade_model)
cv_scaleFactor=1.1
cv_minNeighbors=8
cv_minSize=(64, 64)
cv_flags=cv2.CASCADE_SCALE_IMAGE
resnet_imagesize = 64
resnet = WideResNet(image_size=resnet_imagesize)()
resnet.load_weights(agModel)

def detect_faces(image, margin=0.2):
        """
        :param image: Original image (in opencv BGR) to find faces on
        :param padding: additional margin widht/height percentage
        :return:
            array of face image rectangles {left: 34, top: 11, right: 122, bottom: 232, width:(r-l), height: (b-t)}
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = np.shape(gray)

        face_results = []

        faces = faceCascade.detectMultiScale(gray, 
            scaleFactor=cv_scaleFactor,
            minNeighbors=cv_minNeighbors,
            minSize=cv_minSize,
            flags=cv_flags)
        for (x, y, w, h) in faces:
            xi1 = max(int(x - margin * w), 0)
            xi2 = min(int(x + w + margin * w), img_w - 1)
            yi1 = max(int(y - margin * h), 0)
            yi2 = min(int(y + h + margin * h), img_h - 1)
            detection = {'left': xi1, 'top': yi1, 'right': xi2, 'bottom': yi2,
                         'width': (xi2 - xi1), 'height': (yi2 - yi1)}
            face_results.append(detection)

        return face_results

def aspect_resize(image, width, height, padding=cv2.BORDER_REPLICATE, color=[0, 0, 0]):
    old_size = image.shape[:2]
    ratio = min(float(height) / old_size[0], float(width) / old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])
    im = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = width - new_size[1]
    delta_h = height - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, padding, value=color)
    return new_im


def gender_age(face_image, left=0, top=0, width=None, height=None):
    img_h, img_w, __ = np.shape(face_image)
    if width is not None:
        img_w = width
    if height is not None:
        img_h = height

    test_image = aspect_resize(face_image[top:top + img_h, left:left + img_w], 64,64)
    result = resnet.predict(np.array([test_image]))
    predicted_genders = result[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = result[1].dot(ages).flatten()

    return predicted_genders[0][0], predicted_ages[0]



def detect_genders_ages(image):
    faceregions = detect_faces(image, margin=0.4)
    for face in faceregions:
        face['gender'], face['age'] = gender_age(image,
            left=face['left'], top=face['top'],
            width=face['width'],
            height=face['height'])
    return faceregions

def agegender(image):
    img = cv2.imread(image)
    ag = detect_genders_ages(img)

    return ag

image="02968.png"
print(agegender(image))
