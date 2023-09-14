import cv2

def cam_show():
    video = cv2.VideoCapture(0)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("output.mp4", fourcc, 25, (w, h))
    while (True):
        ok, frame = video.read()
        cv2.imshow('Video', frame)
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()

cam_show()
