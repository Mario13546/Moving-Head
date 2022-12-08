# Created by Alex Pereira

# Import Libraries
import cv2 as cv

# Import Classes
from pose import PoseDetector

# 
class Position:
    def __init__(self, capture, detCon, trackCon) -> None:
        """
        """
        # Reads capture in init
        self.cap = capture
        self.readCapture()
        print("Camera opened sucessfully")

        # Creates an instance of PoseDetector
        self.detector = PoseDetector(detectionCon = detCon, minTrackCon = trackCon)
    
    def readCapture(self):
        """
        Reads the VideoCapture capture.
        @return videoStream
        """
        # Reads the capture
        success, stream = self.cap.read()

        # If read fails, raise an error
        if not success:
            raise OSError("Camera error! Failed to start!")
        
        return stream

    def liveTracking(self):
        """
        Tracks a person's pose. To be used for display purposes.
        @return allDetectedHands
        """
        # Reads the capture
        stream = self.readCapture()

        # Pose detection
        allHands, stream = self.detector.findPose(stream)

        # Show the stream
        cv.imshow("MediaPipe Pose", stream)

        return allHands