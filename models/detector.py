from ultralytics import YOLO
from config import YOLO_MODEL

class ObjectDetector:

    def __init__(self):
        self.model = YOLO(YOLO_MODEL)

    def detect(self, frame):

        results = self.model(frame, verbose=False)

        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls)
            label = self.model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "bbox": (x1, y1, x2, y2)
            })

        return detections