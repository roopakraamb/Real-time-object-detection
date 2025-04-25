import cv2
import numpy as np
from openvino.runtime import Core

# Initialize OpenVINO runtime
ie = Core()

# Define model path
model_path = "C:/Users/BALAJI/OneDrive/Desktop/Real-Time Object Detection/Models/FP32/ssd_mobilenet_v1_coco.xml"

# Load the model
compiled_model = ie.compile_model(model_path, "CPU")

# Get input and output layer names
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'nothing', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'nothing', 'backpack', 'umbrella',
    'nothing', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'nothing', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'nothing',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop for real-time object detection
while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    
    # Check if frame is successfully captured
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize the frame to 300x300 as expected by the model
    frame_resized = cv2.resize(frame, (300, 300))

    # Convert to float32 (OpenVINO models generally expect this)
    input_image = frame_resized.astype(np.float32)

    # Normalize the image (Optional, depends on model)
    # input_image = (input_image - 127.5) / 127.5  # Uncomment if needed

    # Add batch dimension (NHWC format)
    input_image = np.expand_dims(input_image, axis=0)

    # Run inference on the image
    results = compiled_model([input_image])[output_layer]

    # Extract bounding boxes, labels, and scores
    # This depends on the output format of the model
    for detection in results[0][0]:
        confidence = detection[2]
        if confidence > 0.5:  # Show only detections with confidence > 50%
            x_min = int(detection[3] * frame.shape[1])
            y_min = int(detection[4] * frame.shape[0])
            x_max = int(detection[5] * frame.shape[1])
            y_max = int(detection[6] * frame.shape[0])
            label = int(detection[1])
            
            # Get the class name from the class ID
            class_name = COCO_CLASSES[label]
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame with detections
    cv2.imshow("Real-Time Object Detection", frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
