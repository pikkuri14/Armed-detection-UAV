import cv2
import mediapipe as mp

# Initialize MediaPipe image classification module
mp_classification = mp.solutions.classification
classification = mp_classification.Classification(model_selection=1)

# Initialize MediaPipe DrawingUtils for drawing results
mp_drawing = mp.solutions.drawing_utils

# Create a VideoCapture object to capture frames from the webcam
cap = cv2.VideoCapture(0)  # 0 represents the default webcam, change it for multiple cameras

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform image classification
    results = classification.process(rgb_frame)

    if results.classification:
        # Get the top classification result
        top_classification = results.classification[0]

        # Get the label and score
        label = top_classification.label
        score = top_classification.score

        # Display the classification result on the frame
        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Score: {score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with classification result
    cv2.imshow("Image Classification", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
