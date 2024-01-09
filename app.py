import cv2 as cv
import mediapipe as mp
import time
import math

# Calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Classify hand size
def classify_hand_size(scaled_length):
    if 10 <= scaled_length <= 13:
        return "Medium"
    elif scaled_length < 10:
        return "Small"
    else:
        return "Large"

# Open the default camera (camera index 0)
video = cv.VideoCapture(0)

# Initialize the Mediapipe Face and Hands modules
mpFace = mp.solutions.face_detection
face_detection = mpFace.FaceDetection(min_detection_confidence=0.2)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDrawing = mp.solutions.drawing_utils

# Frame rate
pTime = 0
cTime = 0

# Known actual distance between eyes in centimeters  (I don't know if it's the real size... Moshe)
known_actual_eyes_distance_cm = 4

while True:
    success, img = video.read() # Read frame from video capture
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) # Convert the BGR image to RGB
    face_results = face_detection.process(imgRGB) # Process the frame using the Mediapipe Face module to detect faces

    # Extract the landmarks of the first face detected (if any)
    if face_results.detections:
        face_landmarks = face_results.detections[0].location_data.relative_keypoints

        # Calculate the distance between the eyes (known actual distance)
        left_eye = (int(face_landmarks[1].x * img.shape[1]), int(face_landmarks[1].y * img.shape[0]))
        right_eye = (int(face_landmarks[0].x * img.shape[1]), int(face_landmarks[0].y * img.shape[0]))

        eyes_distance_pixels = calculate_distance(left_eye, right_eye)

        # Calculate the scale factor based on the ratio of the measured distance to the known actual distance
        scale_factor = known_actual_eyes_distance_cm / eyes_distance_pixels

        cv.line(img, left_eye, right_eye, (0, 255, 0), 2) # Draw a line between the eyes
        

        results = hands.process(imgRGB) # Process the frame using the Mediapipe Hands module

        # Print the landmarks of detected hands (if any)
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

                # Calculate the distance between index finger tip and wrist
                index_finger_tip = (int(handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * img.shape[1]),
                                    int(handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * img.shape[0]))

                wrist = (int(handLandmarks.landmark[mpHands.HandLandmark.WRIST].x * img.shape[1]),
                         int(handLandmarks.landmark[mpHands.HandLandmark.WRIST].y * img.shape[0]))

                # Calculate the distance in pixels
                length_pixels = calculate_distance(index_finger_tip, wrist)

                # Scale the distance using the previously calculated scale factor
                length_cm = length_pixels * scale_factor

                # Display the scaled length on the frame
                cv.putText(img, f'Scaled Length: {length_cm:.2f} cm', (10, 100), cv.FONT_HERSHEY_PLAIN, 2, (255, 50, 255), 2)

                # Classify hand size based on scaled length
                size_label = classify_hand_size(length_cm)

                # Display the hand size classification
                cv.putText(img, f'Size: {size_label}', (10, 130), cv.FONT_HERSHEY_PLAIN, 2, (255, 50, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display the original frame
    cv.imshow('Video', img)

    # Break the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()

# Close all OpenCV windows
cv.destroyAllWindows()
