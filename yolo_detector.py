import cv2
import torch
from pathlib import Path

# Load YOLO v5 model
def load_yolo_v5_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    model.conf = 0.25  # Confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    # log model.names
    print('Model loaded with %s classes: %s' % (len(model.names), model.names))
    return model

# Process webcam stream and detect cards
def detect_cards_in_webcam_stream(model):
    vid = cv2.VideoCapture(2)
    # print('Webcam resolution: %dx%d' % (vid.get(cv2.CAP_PROP_FRAME_WIDTH), vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # Set vid resolution to 1920x1080
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Detect cards in the frame
        results = model(frame)

        # Draw rectangles around detected cards and display the object name
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            object_name = model.names[int(cls)]
            cv2.putText(frame, object_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the frame
        cv2.imshow('Webcam Stream', frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video resources
    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Download YOLO v5 pre-trained weights (if not already downloaded)
    weights_path = Path('yolov5s.pt')
    if not weights_path.exists():
        print('Downloading YOLO v5 pre-trained weights...')
        torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt', weights_path)

    # Load YOLO v5 model
    print('Loading YOLO v5 model...')
    model = load_yolo_v5_model(weights_path)

    # Process webcam stream and detect cards
    print('Processing webcam stream...')
    detect_cards_in_webcam_stream(model)

    print('Done!')