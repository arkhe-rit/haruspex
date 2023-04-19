import cv2
import threading
import time
from collections import deque

# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name, cv2.CAP_DSHOW )
        print("Frame width:", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Frame height:", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Frames per second:", self.cap.get(cv2.CAP_PROP_FPS))
        print("FourCC codec:", self.cap.get(cv2.CAP_PROP_FOURCC))

        self.q = deque(maxlen=1)  # Set maxlen to 1
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.q.append(frame)

    def read(self):
        return True, self.q.popleft()

    def set(self, prop, value):
        return self.cap.set(prop, value)