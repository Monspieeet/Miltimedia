import requests
import cv2
import numpy as np
import imutils

def cam_show():
    url = "http://192.168.1.134:8080/shot.jpg"
    lower_red1 = np.array([0, 120, 75])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([167, 115, 75])
    upper_red2 = np.array([180, 255, 255])
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        frame = imutils.resize(frame, width=1000, height=1800)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        final_mask = cv2.addWeighted(mask1,0.5, mask2, 0.5, 0.0)
        red_filtered_frame = cv2.bitwise_and(frame, frame, mask=final_mask)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        erosion = cv2.erode(final_mask, kernel, iterations=1)
        dilation = cv2.dilate(final_mask, kernel, iterations=1)

        cv2.imshow("Erosion", erosion)
        cv2.imshow("Dilation", dilation)
        cv2.imshow("Opening", opening)
        cv2.imshow("Closing", closing)
        cv2.imshow('Red Filtered Image', red_filtered_frame)
        cv2.imshow('orig', frame)
        cv2.imshow('hsv', hsv)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


cam_show()
