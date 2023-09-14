import cv2

def cam_show():
    video = cv2.VideoCapture(r'.output3.mp4', cv2.CAP_ANY)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("output.mp4", fourcc, 25, (w, h))

    while (True):
        ok, vid = video.read()
        cv2.imshow('Video', vid)
        video_writer.write(vid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()

cam_show()
