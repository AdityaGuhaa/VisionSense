import cv2
import textwrap

from camera.camera_stream import CameraStream
from models.detector import ObjectDetector
from models.vlm import VisionLLM
from utils.drawing import draw_detections
from utils.fps import FPSCounter

from config import WINDOW_NAME, VLM_INTERVAL


def draw_scene_text(frame, text):

    wrapped = textwrap.wrap(text, width=60)

    y = 80

    for line in wrapped[:3]:   # limit to 3 lines
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 30


def main():

    camera = CameraStream()
    detector = ObjectDetector()
    vlm = VisionLLM()
    fps_counter = FPSCounter()

    frame_count = 0
    scene_text = "Starting system..."

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Camera read failed")
            break

        # Run YOLO detection
        detections = detector.detect(frame)

        # Extract labels
        labels = [d["label"] for d in detections]

        # Draw detection boxes
        frame = draw_detections(frame, detections)

        # Run VLM every N frames
        if frame_count % VLM_INTERVAL == 0:
            scene_text = vlm.describe_scene(frame, labels)

        # Update FPS
        fps = fps_counter.update()

        # Draw FPS
        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Draw scene description
        draw_scene_text(frame, scene_text)

        # Show window
        cv2.imshow(WINDOW_NAME, frame)

        frame_count += 1

        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()