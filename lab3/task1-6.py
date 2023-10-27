import cv2
import numpy as np

def CVBlur(img, kernel_size, deviation):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), deviation)

def GaussBlur(img, kernel_size, standard_deviation):
    kernel = np.ones((kernel_size, kernel_size))
    a = b = (kernel_size + 1) // 2

    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)


    print("//////////")
    sum = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            sum += kernel[i, j]

    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] /= sum

    print(kernel)

    imgBlur = img.copy()
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    for i in range(x_start, imgBlur.shape[0] - x_start):
        for j in range(y_start, imgBlur.shape[1] - y_start):
            val = np.sum(img[i - kernel_size//2: i + kernel_size//2 + 1, j - kernel_size//2: j + kernel_size//2 + 1] * kernel)
            imgBlur[i, j] = val

    return imgBlur


def gauss(x, y, omega, a, b):
    omega2 = 2 * omega ** 2

    m1 = 1 / (np.pi * omega2)
    m2 = np.exp(-((x-a) ** 2 + (y-b) ** 2) / omega2)

    return m1 * m2


def Show():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img = GaussBlur(frame, 11, 50)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

Show()
