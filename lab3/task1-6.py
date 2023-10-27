
import cv2
import numpy as np

def CVBlur(img, kernel_size, deviation):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), deviation)

def GaussBlur(img, kernel_size, standard_deviation):
    kernel = np.ones((kernel_size, kernel_size))
    a = b = (kernel_size + 1) // 2

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



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

    imgBlur = gray_img.copy()
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    for i in range(x_start, imgBlur.shape[0] - x_start):
        for j in range(y_start, imgBlur.shape[1] - y_start):
            val = np.sum(gray_img[i - kernel_size//2: i + kernel_size//2 + 1, j - kernel_size//2: j + kernel_size//2 + 1] * kernel)
            imgBlur[i, j] = val

    for i in range(imgBlur.shape[0]):
        for j in range(imgBlur.shape[1]):
            hsv_img[i,j][2] = imgBlur[i,j]

    imgBlur = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return imgBlur


def gauss(x, y, omega, a, b):
    omega2 = 2 * omega ** 2

    m1 = 1 / (np.pi * omega2)
    m2 = np.exp(-((x-a) ** 2 + (y-b) ** 2) / omega2)

    return m1 * m2


img = cv2.imread('test.jpg')
cv2.imshow('off', img)
cvgauss = CVBlur(img, 5, 100)
mygauss = GaussBlur(img, 5, 100)
cv2.imshow('on', cvgauss)
cv2.imshow('my on', mygauss)
cv2.waitKey(0)
cv2.destroyAllWindows()
