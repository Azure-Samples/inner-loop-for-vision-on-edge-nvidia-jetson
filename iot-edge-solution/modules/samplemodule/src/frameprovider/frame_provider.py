import cv2


class VideoCapture:
    def __init__(self, camera_path, text_detection):
        print("Initializing video capture")
        self.text_detection = text_detection
        self.cap = cv2.VideoCapture(camera_path)
        self.read()

    def read(self):
        print("Capturing video frames")
        while True:
            ret, frame = self.cap.read()
            self.text_detection.run(frame)
