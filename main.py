# Created by Alex Pereira

# Import Libraries
import cv2 as cv

# Import Classes
from head_control import PoseDetector
from camera import USBCamera

# Create a video capture
camera = USBCamera(0)
cap    = camera.getCapture()

# Instance Creation
detector = PoseDetector(1, 0.5, 0.5)

# Main loop
while (cap.isOpened() == True):
    # 
    pass