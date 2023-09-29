import requests
import cv2
import numpy as np
import imutils


def cam_show():
    #url = "http://192.168.211.169:8080/shot.jpg"
    #lower_red1 = np.array([0, 120, 75])
    #upper_red1 = np.array([10, 255, 255])
    #lower_red2 = np.array([167, 115, 75])
    #upper_red2 = np.array([180, 255, 255])
    cap = cv2.VideoCapture(0)
    lower_red1 = np.array([0, 125, 85])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([167, 115, 75])
    upper_red2 = np.array([180, 255, 255])
    while True:
        ret, frame = cap.read()
        #img_resp = requests.get(url)
        #img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        #frame = cv2.imdecode(img_arr, -1)
        #frame = imutils.resize(frame, width=1000, height=1800)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        final_mask = cv2.addWeighted(mask1,1, mask2, 1, 0.0)
        kernel = np.ones((5,5), np.uint8)
        open = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        red_filtered_frame = cv2.bitwise_and(frame, frame, mask=open)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > 300:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 0), 2)

        cv2.imshow('Red Filtered Image', red_filtered_frame)
        cv2.imshow('orig', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


cam_show()
