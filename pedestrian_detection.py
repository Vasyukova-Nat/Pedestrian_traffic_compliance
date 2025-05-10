import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from config import YOLO_MODEL_PATH, VIDEO_PATH, OUTPUT_DETECTION_FILE


prev_positions = {}

def detect(frame, results):
    detections = sv.Detections.from_ultralytics(results)  # конвертация в формат supervision  
    detections = sv.ByteTrack().update_with_detections(detections)
    detections = detections[np.isin(detections.class_id, [0])]  # фильтруем только пешеходов (id==0)

    speeds = []
    for detection in detections:
        xyxy = detection[0]  # Bounding box coordinates [x1,y1,x2,y2]
        tracker_id = detection[4]  
        
        center = ((xyxy[0] + xyxy[2])/2, (xyxy[1] + xyxy[3])/2)
        
        # Расчет скорости пешехода (пикселей/кадр)
        if tracker_id in prev_positions:
            prev_center = prev_positions[tracker_id]
            speed = np.sqrt((center[0]-prev_center[0])**2 + (center[1]-prev_center[1])**2)
            speeds.append(speed)
        else:
            speeds.append(0)
        
        prev_positions[tracker_id] = center
    
    return detections, np.array(speeds)

def annotate_frame(frame, detections, speeds):
    # First annotate (отрисовка) bounding boxes
    annotated_frame = sv.BoxAnnotator().annotate(
        scene=frame.copy(),
        detections=detections
    )

    # Prepare labels
    labels = [
        f"{'Moving' if speed > 2 else 'Standing'} {speed:.1f}px/frame"
        for speed in speeds
    ]

    # Draw labels
    annotated_frame = sv.LabelAnnotator().annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # Pedestrian counter
    cv2.putText(annotated_frame,
               f"Pedestrians: {len(detections)}",
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (0, 255, 0), 2)

    return annotated_frame

def PedestrianDetector():
    model = YOLO(YOLO_MODEL_PATH)

    print(f"Processing video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл.")
        return
    
    # Получение свойств видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    
    print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps:.1f}")
    
    out = cv2.VideoWriter(OUTPUT_DETECTION_FILE, 
                         cv2.VideoWriter_fourcc(*'mp4v'),  # кодек MP4V
                         fps, 
                         (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детекция людей. В COCO человек - это класс person, его ID==0
        results = model(frame, classes=[0], verbose=False)

        detections, speeds = detect(frame, results[0]) 

        annotated_frame = annotate_frame(frame, detections, speeds)
        
        cv2.imshow('Pedestrian Detection (Press ESC to quit)', annotated_frame)
        out.write(annotated_frame)
        if cv2.waitKey(1) == 27:  # выход по клавише ESC, её код 27
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Видео обработано. Результат сохранен в {OUTPUT_DETECTION_FILE}")

if __name__ == "__main__":
    PedestrianDetector()