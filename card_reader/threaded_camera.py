import cv2
from threading import Thread
import time

class ThreadedCamera():
    def __init__(self, src=0, *rest):
        self.capture = cv2.VideoCapture(src, *rest)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        
        self.frame = None

        FPS = 30
        self.FPS_S = 1/FPS

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            try:
                if self.capture.isOpened():
                    (self.status, self.frame) = self.capture.read()
                time.sleep(self.FPS_S)
            except cv2.error as e:
                print(f"Error reading frame from camera: {e}")
                time.sleep(10 * self.FPS_S)

    
    def set(self, *args):
        return self.capture.set(*args)
    
    def latest(self):
        while self.frame is None:
            time.sleep(self.FPS_S)
        return self.frame