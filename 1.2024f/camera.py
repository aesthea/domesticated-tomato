import cv2
import numpy as np
import subprocess
import math

class camera:
    def __init__(self):
        self.device_array = []
        self.device_index = None
        for i in range(10):
            self.cap = cv2.VideoCapture(i)
            if not self.cap.read()[0]:
                continue
            else:
                self.device_array.append(i)
            self.cap.release()
        print("device found at : ", self.device_array)
    
    def connect(self, device_index):
        self.cap = cv2.VideoCapture(device_index)
        self.device_index = device_index

    def close(self):
        self.cap.release()
        self.device_index = None

    def read(self):
        ret, im = self.cap.read()
        return ret, im


class dinolite:
    def __init__(self, DINOLITE_COM = 0):
        self.led = subprocess.Popen(r"LED\LED_AE.exe",shell = False)
        print(self.led.pid)
        self.DINOLITE_COM = DINOLITE_COM
        self.cam = cv2.VideoCapture(self.DINOLITE_COM)
        self.cam.set(3, 640)
        self.cam.set(4, 480)
        #self.cam.set(3, 1280)
        #self.cam.set(4, 960)
        self.cam.set(5, 30)
    def capture(self):
        h_shift_perc = 0.0
        try:
            ret, frame = self.cam.read()
            if frame.shape[0] > 100:
                frame = np.rot90(frame, k = 2)
            else:
                frame = np.zeros((480,480,3))
                frame = frame.astype("uint8")
                print("ERROR CAPTURE")
        except Exception as e:
            frame = np.zeros((480,480,3))
            frame = frame.astype("uint8")
            print("EXCEPTION 1645," , e)
        h,w,d = frame.shape
        if w > h:
            diff = math.floor((w - h)/2)
            w_shift = int(math.floor(diff * h_shift_perc))
            frame = frame[0:h, diff + w_shift: (w - diff) + w_shift]
        elif h > w:
            diff = math.floor((h - w)/2)
            h_shift = int(math.floor(diff * h_shift_perc))
            frame = frame[diff + h_shft: (h - diff) + h_shift, 0: w]
        return frame
    def close(self):
        cv2.VideoCapture(self.DINOLITE_COM).release()
        del self.cam
