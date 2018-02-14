import numpy as np
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
from findDerivatives import findDerivatives


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def filter(I, n):
    if n == "3":
        I = I.astype(int)
        Ig = rgb2gray(I)
        Ig = Ig.astype(np.float64())
        [gradx, grady] = np.gradient(Ig)
        e = np.abs(gradx) + np.abs(grady)
        kernel = np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        I[e > e * 0.5, 0] = signal.convolve2d(I[:, :, 0], kernel, 'same')[e > e * 0.5]
        I[e > e * 0.5, 1] = signal.convolve2d(I[:, :, 1], kernel, 'same')[e > e * 0.5]
        I[e > e * 0.5, 2] = signal.convolve2d(I[:, :, 2], kernel, 'same')[e > e * 0.5]
        I[I > 255] = 255
        I[I < 0] = 0
        I = I.astype(np.uint8)
        # I = cv2.GaussianBlur(I, (3, 3), 0)
        return I
    elif n == "2":
        ker = 10
        I = cv2.morphologyEx(I, cv2.MORPH_CLOSE, np.ones((ker, ker), dtype='uint8'))
        I = I.astype(np.uint8)
        return I
    elif n == "1":
        I = I.astype(np.int)
        Ig = rgb2gray(I)
        Ig = Ig.astype(np.float64())
        [gradx, grady] = np.gradient(Ig)
        e = np.abs(gradx) + np.abs(grady)
        mag = findDerivatives(Ig)
        I[mag > mag * 0.9, 0] = I[mag > mag * 0.9, 0] * 1.2
        I[mag > mag * 0.9, 1] = I[mag > mag * 0.9, 1] * 1.2
        I[mag > mag * 0.9, 2] = I[mag > mag * 0.9, 2] * 1.2
        I[I > 255] = 255
        I[I < 0] = 0
        I = I.astype(np.uint8)
        return I
    elif n == "4":
        I = cv2.GaussianBlur(I, (15, 15), 0)
        return I
    elif n == "5":
        newI = np.zeros(np.shape(I))
        I = I.astype(int)
        IR = I[:, :, 0] * random.randint(50, 100) / 100
        IG = I[:, :, 1] * random.randint(50, 100) / 100
        IB = I[:, :, 2] * random.randint(50, 100) / 100
        newI[:, :, 0] = IR
        newI[:, :, 1] = IG
        newI[:, :, 2] = IB
        newI = newI.astype('uint8')
        return newI


if __name__ == "__main__":
    image_path = './images/16.jpg'
    I = np.array(Image.open(image_path).convert('RGB'))
    ori = I.copy()
    choice = ""
    while 1:
        choice = input(
            'Choose a filter number:\n 1.enhance light \n 2.painting \n 3.sharpen \n 4.blur \n 5.magic color \n'
            'Please choose a number:\n'
            'press Q to quit\n')
        if choice == "Q":
            break
        newI = filter(I, choice)
        fig, (a1, a2) = plt.subplots(1, 2)
        a1.imshow(ori)
        a1.axis("off")
        a2.imshow(newI)
        a2.axis("off")
        plt.show()
