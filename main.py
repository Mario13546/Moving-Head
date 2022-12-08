# Created by Alex Pereira

# Import Libraries
import cv2 as cv

# Import Classes
from camera import USBCamera
from head_control import Movement

# Create a video capture
camera = USBCamera(0)
cap    = camera.getCapture()

# Instance Creation
move = Movement(cap, 0.75, 0.75)

# Main loop
while (cap.isOpened() == True):
    # Execution
    move.liveTracking()

    # Press q to end the program
    if ( cv.waitKey(1) == ord("q") ):
        print("Process Ended by User")
        cv.destroyAllWindows()
        cap.release()
        break