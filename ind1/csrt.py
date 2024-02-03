import cv2
import numpy as np
from pathlib import Path

# paths
CURRENT_DIR = Path(__file__).parent
MEDIA_DIR = CURRENT_DIR.parent / Path("media")

# Load the video
video = cv2.VideoCapture(str("hand.mp4"))

# Initialize the KCF tracker
tracker = cv2.TrackerCSRT_create()

# Read the first frame
ret, frame = video.read()

# Initialize the bounding box of the object to track
bbox = cv2.selectROI("Tracking", frame, False)

# Start the tracking process
success = tracker.init(frame, bbox)

while True:
    # Read the next frame
    ret, frame = video.read()

    # Exit if the video has ended
    if not ret:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)

    # Draw the bounding box of the tracked object
    if success:
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break

# Release the video and destroy windows
video.release()
cv2.destroyAllWindows()
