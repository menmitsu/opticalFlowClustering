import cv2 as cv
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt

class ComputeOpticalFLow:
    def __init__(self, firstframe):
        self.firstframe = firstframe
        self.width = self.firstframe.shape[1]
        self.height = self.firstframe.shape[0]

        self.outputImg = np.zeros([self.height, 2*self.width, 3], 
                                  dtype=self.firstframe.dtype)
        self.mask = np.zeros_like(self.firstframe)
        self.mask[..., 1] = 255
        self.prev_gray = cv.cvtColor(self.firstframe, cv.COLOR_BGR2GRAY)
    
    def compute(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(self.prev_gray, gray,
									None,
									0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Sets image hue according to the optical flow direction
        self.mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow magnitude (normalized)
        self.mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        rgb = cv.cvtColor(self.mask, cv.COLOR_HSV2BGR)
        self.prev_gray = gray

        return rgb

if __name__ == '__main__':
    vid_path = 'video_lq.mp4' 
    cap = cv.VideoCapture(vid_path)
    ret, firstframe = cap.read()
    compflow = ComputeOpticalFLow(firstframe)

    while (cap.isOpened()):
        ret, frame = cap.read()
        opflowimg = compflow.compute(frame)

        cv.imshow('Win', opflowimg)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break