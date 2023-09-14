import cv2
import numpy as np


def cam_show():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        height, width, _ = frame.shape
        cross_image = np.zeros((height, width, 3), dtype=np.uint8)
        vertical_line_width = 60
        vertical_line_height = 300
        cv2.rectangle(cross_image,
                      (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2),
                      (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2),
                      (0, 0, 255), 2)
        rect_start_v = (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2)
        rect_end_v = (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2)
        horizontal_line_width = 250
        horizontal_line_height = 55
        cv2.rectangle(cross_image,
                      (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2),
                      (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2),
                      (0, 0, 255), 2)

        rect_start_h = (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2)
        rect_end_h = (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2)
        central_pixel_color = frame[height // 2, width // 2]
        color_distances = [
            np.linalg.norm(central_pixel_color - np.array([0, 0, 255])),
            np.linalg.norm(central_pixel_color - np.array([0, 255, 0])),
            np.linalg.norm(central_pixel_color - np.array([255, 0, 0]))
        ]
        closest_color_index = np.argmin(color_distances)

        if closest_color_index == 0:
            cv2.rectangle(cross_image, rect_start_h, rect_end_h, (0, 0, 255), -1)
        elif closest_color_index == 1:
            cv2.rectangle(cross_image, rect_start_h, rect_end_h, (0, 255, 0), -1)
        else:
            cv2.rectangle(cross_image, rect_start_h, rect_end_h, (255, 0, 0), -1)
        if closest_color_index == 0:
            cv2.rectangle(cross_image, rect_start_v, rect_end_v, (0, 0, 255), -1)
        elif closest_color_index == 1:
            cv2.rectangle(cross_image, rect_start_v, rect_end_v, (0, 255, 0), -1)
        else:
            cv2.rectangle(cross_image, rect_start_v, rect_end_v, (255, 0, 0), -1)
        result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)
        cv2.imshow('frame', result_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

cam_show()
