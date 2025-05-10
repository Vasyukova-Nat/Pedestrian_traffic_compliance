import cv2
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, VIDEO_PATH

def PedestrianDetector():
    model = YOLO(YOLO_MODEL_PATH)

    print(f"Processing video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл.")
        return
    
    # получение свойств видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    
    print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps:.1f}")
    
    out = cv2.VideoWriter('output.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),  # кодек MP4V
                         fps, 
                         (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детекция людей. В COCO человек - это класс person, его ID==0
        results = model(frame, classes=[0], verbose=False) # conf - порог уверенности

        annotated_frame = results[0].plot() # Визуализация
        
        cv2.imshow('Pedestrian Detection (Press ESC to quit)', annotated_frame)
        out.write(annotated_frame) # Сохранение результата
        if cv2.waitKey(1) == 27:  # выход по нажатию клавиши ESC, её код 27
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    PedestrianDetector()