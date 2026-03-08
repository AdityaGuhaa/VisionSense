import cv2

from camera.camera_stream import CameraStream
from models.detector import ObjectDetector
from models.vlm import VisionLLM
from utils.drawing import draw_detections
from utils.fps import FPSCounter

from config import WINDOW_NAME, VLM_INTERVAL

camera = CameraStream()
detector = ObjectDetector()
vlm = VisionLLM()
fps_counter = FPSCounter()

frame_count = 0
scene_text = ""

while True:

    ret, frame = camera.read()

    if not ret:
        break

    detections = detector.detect(frame)

    labels = [d["label"] for d in detections]

    frame = draw_detections(frame, detections)

    if frame_count % VLM_INTERVAL == 0:
        scene_text = vlm.describe_scene(labels)

    fps = fps_counter.update()

    cv2.putText(frame, f"FPS: {fps}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame, scene_text, (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.imshow(WINDOW_NAME, frame)

    frame_count += 1

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()