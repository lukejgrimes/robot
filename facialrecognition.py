import cv2
import dlib
from keras_facenet import FaceNet

class FacialRecognition:
    def __init__(self):
        self.embedder = FaceNet() # Initalizes FaceNet Model

    def recognize_face(self):
        image, boundries = self.get_user_image() # Returns an image of the user and the boundries that contain their face
        detections = self.embedder.extract(image, threshold=0.95) # Model creates an embedding from the image along with some facial features
        return detections[0]['embedding'].tolist() # Return just the embedding as a list

    def get_user_image(self):
        face_detected = False
        cap = cv2.VideoCapture(0) # Open camera
        detector = dlib.get_frontal_face_detector() # Initialize the face detector

        while not face_detected:
            # Capture images from the camera until a face has been detected
            ret, frame = cap.read() 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the image to a RGB color scheme
            dets = detector(frame) # Check frame for a face
            for i, d in enumerate(dets):
                # Once the face has been detected, set the boundries
                face_detected = True
                boundries = [d.left(), d.top(), d.right(), d.bottom()]

        cap.release() # Close camera
        # Return image of face and the boundries
        return frame, boundries
    