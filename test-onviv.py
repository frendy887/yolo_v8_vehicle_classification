from ultralytics import YOLO
from imutils.video import VideoStream
import cv2
import imutils
import os


# RTSP URL for your camera
RTSP_URL_HD = "rtsp://admin:admiqn@192.168.0.100:8554/Streaming/Channels/101"
RTSP_URL_SD = "rtsp://admin:admin@192.168.0.100:8554/Streaming/Channels/102"

# Initialize video capture from the RTSP stream
cap = cv2.VideoCapture(RTSP_URL_HD)


model_path = os.path.join('.', 'yolov8n.pt')
model = YOLO(model_path)  # Load a custom YOLO model
threshold = 0.20

try:
    # Read and display frames in a loop
    while True:
        ret, frame = cap.read()
        print(f"frame size {frame.shape}")
        if frame is None:
            print("Error: Failed to read frame from stream.")
            break


        results = model(frame)[0]
        # Draw bounding boxes and labels for detections
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper() + " " + str(round(score, 2)), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

        # Display the frame
        # frame = imutils.resize(frame,width=720)
        cv2.imshow("Camera Feed", frame)

        # Exit when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

except Exception as e :
    print(f'Error cctv: {e}')

