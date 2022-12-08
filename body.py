# Created by Alex Pereira

# Import Libraries
import cv2 as cv
import mediapipe as mp

# Creates the BodyDetector Class
class BodyDetector:
    def __init__(self, detectionCon = 0.5, minTrackCon = 0.5):
        """
        Constructor for BodyDetector.
        @param detectionConfidence
        @param minimumTrackingConfidence
        """
        # Initiaizes the MediaPipe Holistic solution
        self.mpHolistic = mp.solutions.holistic
        self.holistic   = self.mpHolistic.Holistic( static_image_mode = False,
                                                    model_complexity = 1,
                                                    smooth_landmarks = True,
                                                    enable_segmentation = True,
                                                    smooth_segmentation = True,
                                                    refine_face_landmarks = True,
                                                    min_detection_confidence = detectionCon,
                                                    min_tracking_confidence = minTrackCon)

        # Creates the drawing objects
        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Creates variables
        self.count = 0

    def findBody(self, stream):
        """
        Finds a person in a stream.
        @param stream
        @return annotatedStream
        """
        # Flips the image
        stream = cv.flip(stream, 1)

        # Converts the image to RGB
        streamRGB = cv.cvtColor(stream, cv.COLOR_BGR2RGB)
        self.results = self.holistic.process(streamRGB)
        stream.flags.writeable = False

        # Gets the shape of the image
        self.height, self.width, self.center = stream.shape

        # If the pose_landmarks or face_landmarks array is not empty
        if ((self.results.pose_landmarks is not None) or (self.results.face_landmarks is not None)):
            # Lets the pose solution build a buffer
            if (self.count != 0):
                # Generates some empty lists
                myLandmarkList, xList, yList = [], [], []

                # Checks if the pose list exists
                if (self.results.pose_landmarks is not None):
                    # Loops through detected pose landmarks
                    for ind, landmark in enumerate(self.results.pose_landmarks.landmark):
                        # Adds cordinates to the landmarkList
                        px, py, pz = int(landmark.x * self.width), int(landmark.y * self.height), int(landmark.z * self.width)
                        myLandmarkList.append([px, py, pz, ind])
                        xList.append(px)
                        yList.append(py)
                
                # Checks if the face list exists
                if (self.results.face_landmarks is not None):
                    # Loops through detected face landmarks
                    for ind, landmark in enumerate(self.results.face_landmarks.landmark):
                        # Adds cordinates to the landmarkList
                        px, py, pz = int(landmark.x * self.width), int(landmark.y * self.height), int(landmark.z * self.width)
                        myLandmarkList.append([px, py, pz, ind])
                        xList.append(px)
                        yList.append(py)

                # Bounding Box
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                boundingBox = xmin, ymin, boxW, boxH
                cx, cy = boundingBox[0] + (boundingBox[2] / 2), boundingBox[1] + (boundingBox[3] / 2)

                # Sets the stream to writable
                stream.flags.writeable = True

                # Draws the face landmarks
                self.mp_drawing.draw_landmarks( stream,
                                                self.results.face_landmarks,
                                                self.mpHolistic.FACEMESH_TESSELATION,
                                                landmark_drawing_spec = None,
                                                connection_drawing_spec = self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # Draws the Pose landmarks
                self.mp_drawing.draw_landmarks( stream,
                                                self.results.pose_landmarks,
                                                self.mpHolistic.POSE_CONNECTIONS,
                                                landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style())

                # Draw the bounding box
                cv.rectangle(stream, (boundingBox[0] - 20, boundingBox[1] - 20),
                                    (boundingBox[0] + boundingBox[2] + 20, boundingBox[1] + boundingBox[3] + 20),
                                        (255, 0, 255), 2)

                # Draws on corners to the rectangle because they're cool
                colorC    = (0, 255, 0)
                l, t, adj = 30, 5, 20
                xmin, ymin = xmin - adj, ymin - adj
                xmax, ymax = xmax + adj, ymax + adj
                cv.line(stream, (xmin, ymax), (xmin + l, ymax), colorC, t) # Bottom Left   (xmin, ymax)
                cv.line(stream, (xmin, ymax), (xmin, ymax - l), colorC, t) # Bottom Left   (xmin, ymax)
                cv.line(stream, (xmax, ymax), (xmax - l, ymax), colorC, t) # Bottom Right  (xmax, ymax)
                cv.line(stream, (xmax, ymax), (xmax, ymax - l), colorC, t) # Bottom Right  (xmax, ymax)
                cv.line(stream, (xmin, ymin), (xmin + l, ymin), colorC, t) # Top Left      (xmin, ymin)
                cv.line(stream, (xmin, ymin), (xmin, ymin + l), colorC, t) # Top Left      (xmin, ymin)
                cv.line(stream, (xmax, ymin), (xmax - l, ymin), colorC, t) # Top Right     (xmax, ymin)
                cv.line(stream, (xmax, ymin), (xmax, ymin + l), colorC, t) # Top Right     (xmax, ymin)
            else:
                # Makes cx and cy out of bounds
                cx, cy = -1, -1
        else:
            # Makes cx and cy out of bounds
            cx, cy = -1, -1
        
        # Makes a center point
        center = (cx, cy)
        
        # Increments count
        self.count += 1

        return stream, center