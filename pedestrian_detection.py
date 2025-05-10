import os
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from config import YOLO_MODEL_PATH, VIDEO_PATH

class PedestrianDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        self.class_ids = [0] # В COCO человек - это класс person, его ID==0
        self.tracker = sv.ByteTrack()  # Трекер объектов
        self.box_annotator = sv.BoxAnnotator()  # отрисовка bounding boxes
        self.label_annotator = sv.LabelAnnotator()  # отрисовка меток
        self.prev_positions = {}  # хранение предыдущих позиций

    def detect(self, frame):
        results = self.model(frame, imgsz=640, verbose=False)[0]  # детекция объектов
        detections = sv.Detections.from_ultralytics(results)  # конвертация в формат supervision
        detections = self.tracker.update_with_detections(detections)
        detections = detections[np.isin(detections.class_id, self.class_ids)]  # фильтруем только пешеходов (id==0)

        speeds = []
        for detection in detections:
            xyxy = detection[0]  # Bounding box coordinates [x1,y1,x2,y2]
            tracker_id = detection[4]

            center = ((xyxy[0] + xyxy[2])/2, (xyxy[1] + xyxy[3])/2)

            # Расчет скорости пешехода (пикселей/кадр)
            if tracker_id in self.prev_positions:
                prev_center = self.prev_positions[tracker_id]
                speed = np.sqrt((center[0]-prev_center[0])**2 + (center[1]-prev_center[1])**2)
                speeds.append(speed)
            else:
                speeds.append(0)

            self.prev_positions[tracker_id] = center

        return detections, np.array(speeds)

    def annotate_frame(self, frame, detections, speeds):
        # First annotate (отрисовка) bounding boxes
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        # Prepare labels (метки)
        labels = [
            f"{'Moving' if speed > 2 else 'Standing'} {speed:.1f}px/frame"
            for speed in speeds
        ]

        # отрисовка меток
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        # pedestrian counter (счетчик)
        cv2.putText(annotated_frame,
                   f"Pedestrians: {len(detections)}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

        return annotated_frame

if not os.path.exists(VIDEO_PATH):
    print(f"Error: File {VIDEO_PATH} not found!")
    print("Please place the video file in the same directory as this script")

print("Initializing detector...")
detector = PedestrianDetector()

try:
    print(f"Processing video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video file")

    # получение свойств видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    # delay = int(1000 / fps) if fps > 0 else 25  # расчет задержки (скорости проигрывания)
    delay = 1 # ручная, высокая скорость
    
    print(f"Video resolution: {width}x{height}, FPS: {fps:.1f}")

    while True: 
        ret, frame = cap.read()
        if not ret: 
            print("End of video or error reading frame")
            break

        # Изменение размера кадра (опционально)
        frame = cv2.resize(frame, (1280, 720))

        detections, speeds = detector.detect(frame)
        annotated_frame = detector.annotate_frame(frame, detections, speeds)

        cv2.imshow("Pedestrian Detection (Press ESC to quit)", annotated_frame)

        key = cv2.waitKey(delay) & 0xFF  # Ожидание нажатия клавиши (задержка delay)
        if key == 27:  # Код клавиши ESC
            print("ESC pressed - exiting")
            break
        if cv2.getWindowProperty("Pedestrian Detection (Press ESC to quit)", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed - exiting")
            break

except Exception as e:
    print(f"Error: {str(e)}")
finally:
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    print("Processing completed")
