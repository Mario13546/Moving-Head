# Created by Alex Pereira

# Import Libraries
import cv2 as cv

# Import Classes
from body import BodyDetector
from serial_communication import SerialComms

# Creates the Movement Class
class Movement:
    def __init__(self, capture, detCon, trackCon) -> None:
        """
        """
        # Reads capture in init
        self.cap = capture
        self.readCapture()
        print("Camera opened sucessfully")

        # Creates an instance of BodyDetector
        self.detector = BodyDetector(detectionCon = detCon, minTrackCon = trackCon)
        
        # Creates a SerialComms instance
        self.ser = SerialComms()
    
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
        Tracks a person in the frame. Useful for display purposes.
        @return allDetectedHands
        """
        # Reads the capture
        stream = self.readCapture()

        # Pose detection
        stream, center = self.detector.findBody(stream)

        # Send to servo
        self.sendPosition(stream, center)

        # Show the stream
        cv.imshow("MediaPipe Pose", stream)
    
    def sendPosition(self, stream, bbCenter: tuple):
        """
        Sends the position data to the servo.
        @param stream
        @param center (x, y)
        """
        # Sets min and max positions
        MIN = 0
        MAX = 78

        # Gets the height and width of the stream
        height, width, streamCenter = stream.shape

        # Gets the center point of the bounding box
        cx, cy = bbCenter

        # Normailzes the camera data to the servo range
        data = (cx * (MAX / width))

        # Removes extrenuous values
        if (data == -1 or data == 91):
            # Passes if the two error codes are returned
            pass
        elif (data < MIN):
            # Makes the lowest value the MIN
            data = MIN
        elif (data > MAX):
            # Makes the highest value the MAX
            data = MAX

        # Sends the data to the servo
        myString = self.ser.sendData(data)

        # Prints the encoded data
        print(myString)