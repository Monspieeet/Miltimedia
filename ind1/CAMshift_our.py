import cv2
import numpy as np
from pathlib import Path

def updateHistograms(imgBGR, curWindow):

        bgrObjectRoi = imgBGR[curWindow[1]: curWindow[1] + curWindow[3],
                                   curWindow[0]: curWindow[0] + curWindow[2]]
        hsvObjectRoi = cv2.cvtColor(bgrObjectRoi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsvObjectRoi,
                                np.array((0., 50., 50.)),
                                np.array((180, 255., 255.)))

        histObjectRoi = cv2.calcHist(
            [hsvObjectRoi], [0], mask, [180], [0, 180])
        return cv2.normalize(histObjectRoi, histObjectRoi,
                      0, 255, cv2.NORM_MINMAX)
        


CURRENT_DIR = Path(__file__).parent
MEDIA_DIR = CURRENT_DIR.parent / Path("media")

video = cv2.VideoCapture("hand.mp4")

ok, frame = video.read()

bbox = cv2.selectROI("Tracking", frame, False)

term_criteria = (cv2.TERM_CRITERIA_EPS |
                 cv2.TERM_CRITERIA_COUNT, 10, 1)

currentWindow = bbox
histObjectRoi = updateHistograms(frame, currentWindow)
while True:
    ok, frame = video.read()
    if not ok:
        break

    timer = cv2.getTickCount()
    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    backProjectedImg = cv2.calcBackProject(
        [imgHSV], [0], histObjectRoi, [0, 180], 1)

    backProjectedImg = backProjectedImg

    backProjectedImg_copy = backProjectedImg.copy()

    rotatedWindow, curWindow = cv2.CamShift(
        backProjectedImg, currentWindow, term_criteria)

    rotatedWindow = cv2.boxPoints(rotatedWindow)
    rotatedWindow = np.int0(rotatedWindow)

    currentWindow = curWindow
    
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    x, y, w, h = currentWindow

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)

    rotatedWindow = rotatedWindow

    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    cv2.imshow("CAMShift Tracking", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
