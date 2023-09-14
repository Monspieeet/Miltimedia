import cv2

def cam_show():
    video = cv2.VideoCapture(0)

    while (True):
        ok, frame = video.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        cv2.imshow('Hsv', hsv)
        cv2.imshow('original', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()

cam_show()
