# from imageai.Detection import ObjectDetection
# import cv2
#
# # Initialize the object detection model
# detector = ObjectDetection()
#
# # Set the model to use (e.g., 'YOLOv3' or 'RetinaNet')
# model_path = "tiny-yolov3.pt"
# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath(model_path)
# detector.loadModel()
#
# # Create a VideoCapture object to capture frames from the webcam
# video_capture = cv2.VideoCapture(1)  # 0 represents the default webcam, change it for multiple cameras
#
# while True:
#     # Read a frame from the webcam
#     ret, frame = video_capture.read()
#
#     if not ret:
#         break
#
#     # Perform object detection on the frame
#     detections = detector.detectObjectsFromImage(
#         input_image=frame,
#         output_type="array",
#         display_percentage_probability=True,
#         display_object_name=True
#     )
#
#     # Process the detections and draw bounding boxes and labels
#     for detection in detections[1]:
#         name = detection["name"]
#         percentage_probability = detection["percentage_probability"]
#         box_points = detection["box_points"]
#
#         # Convert box_points to integers for indexing
#         box_points = [int(val) for val in box_points]
#
#         # Draw bounding boxes
#         cv2.rectangle(frame, (box_points[0], box_points[1]), (box_points[2], box_points[3]), (0, 0, 255), 2)
#
#         # Add labels
#         label = f"{name} ({percentage_probability:.2f}%)"
#         cv2.putText(frame, label, (box_points[0], box_points[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#     # Display the frame with bounding boxes and labels
#     cv2.imshow("Object Detection", frame)
#
#     # Exit the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# # Release the video capture and close the OpenCV window
# video_capture.release()
# cv2.destroyAllWindows()
### video detection bisaaaa



# from imageai.Detection import VideoObjectDetection
# import os
# import cv2
#
# def forFrame(frame_number, output_array, output_count):
#     print("FOR FRAME " , frame_number)
#     print("Output for each object : ", output_array)
#     print("Output count for unique objects : ", output_count)
#     print("------------END OF A FRAME --------------")
#
# execution_path = os.getcwd()
#
#
# camera = cv2.VideoCapture(1)
#
# detector = VideoObjectDetection()
# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath(os.path.join(execution_path , "tiny-yolov3.pt"))
# detector.loadModel()
#
#
# video_path = detector.detectObjectsFromVideo(
#                 camera_input=camera,
#                 per_frame_function=forFrame,
#                 output_file_path=os.path.join(execution_path, "camera_detected_video"),
#                 frames_per_second=20, log_progress=True, minimum_percentage_probability=40
# )
#
# print(video_path)
#bisa juga


from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture("IMG_4270.mov")  # Initialize video capture
maskClassifier = Classifier('model_lapang/keras_model.h5', 'model_lapang/labels.txt')

while True:
    _, img = cap.read()  # Capture frame-by-frame
    prediction = maskClassifier.getPrediction(img)
    print(prediction)  # Print prediction result
    cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF

    # Check if the key pressed is 'q' or 'Q' to exit the loop
    if key == ord('q') or key == ord('Q'):
        break


#with the gps
# from cvzone.ClassificationModule import Classifier
# import cv2
# import threading
# from pymavlink import mavutil
#
# def get_current_gps_coordinates(master):
#     while True:
#         msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
#         if msg is not None:
#             lat = msg.lat / 1e7  # Convert to degrees
#             lon = msg.lon / 1e7  # Convert to degrees
#             alt = msg.alt / 1e3   # Convert to meters
#             print(f"Latitude: {lat}, Longitude: {lon}, Altitude: {alt} meters")
#
# def main():
#     cap = cv2.VideoCapture(1)  # Initialize video capture
#     maskClassifier = Classifier('model/keras_model.h5')
#
#     connection_string = 'serial:COM25:57600'
#
#     try:
#         master = mavutil.mavlink_connection('COM25', 57600)
#         print("Connected to autopilot")
#
#         # Start a thread to continuously get GPS coordinates
#         gps_thread = threading.Thread(target=get_current_gps_coordinates, args=(master,))
#         gps_thread.daemon = True
#         gps_thread.start()
#
#         while True:
#             # Add your other tasks here or simply keep the main thread alive
#             _, img = cap.read()  # Capture frame-by-frame
#             prediction = maskClassifier.getPrediction(img)
#             print(prediction)  # Print prediction result
#             cv2.imshow("Image", img)
#             cv2.waitKey(1)  # Wait for a key press
#
#     except Exception as e:
#         print(f"Error: {str(e)}")
#
#
# if __name__ == "__main__":
#     main()



