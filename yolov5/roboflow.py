from roboflow import Roboflow
rf = Roboflow(api_key="i11Ed7fwE3tqL67G4FF2")
project = rf.workspace().project("rifle-6srh6")
model = project.version(2).model

# infer on a local image
# print(model.predict("asdd.jpeg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("asd.mp4", confidence=40, overlap=30).save("prediction.mp4")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())


import cv2
import requests
import numpy as np

# Access the webcam (usually webcam 0)
cap = cv2.VideoCapture(0)

# Set your Roboflow API endpoint and API key
api_endpoint = "https://api.roboflow.com/YOUR_MODEL_ID/infer"
api_key = "i11Ed7fwE3tqL67G4FF2"

while True:
    ret, frame = cap.read()

    # Convert the frame to bytes
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    files = {
        'file': ('frame.jpg', img_bytes)
    }

    # Send a POST request to the Roboflow API for object detection
    response = requests.post(api_endpoint, headers=headers, files=files)

    # Process the API response
    if response.status_code == 200:
        detection_results = response.json()
        for result in detection_results:
            class_name = result['class']
            confidence = result['confidence']
            box = result['boundingBox']

            x = int(box['left'] * frame.shape[1])
            y = int(box['top'] * frame.shape[0])
            width = int(box['width'] * frame.shape[1])
            height = int(box['height'] * frame.shape[0])

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



