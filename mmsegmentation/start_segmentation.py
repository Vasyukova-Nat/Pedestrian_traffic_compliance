import cv2
import numpy as np
from mmseg.apis import inference_model, init_model
from mmseg.utils import get_palette
import time

show_window = 0 # Настройка отображения окна (1 - показывать, 0 - скрыть)

VIDEO_PATH = "selenium_video_11s.mp4"
# VIDEO_PATH = "video_11s.mp4"
# Конфигурация модели и путь к весам
config_file = 'configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth'
# config_file = 'configs/deeplabv3plus/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.py'
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth'

# Инициализация модели
print("Инициализация модели...")
start_time = time.time()
model = init_model(config_file, checkpoint_file, device='cpu')
print(f"Модель загружена за {time.time() - start_time:.2f} секунд")

# Палитра цветов для Cityscapes (19 классов)
palette = get_palette('cityscapes')

# Определяем индекс класса "sidewalk" (тротуар) в Cityscapes
# Классы Cityscapes: 0:road, 1:sidewalk, 2:building, ..., 18:license plate
SIDEWALK_CLASS_ID = 1

# Открытие видеофайла
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Ошибка открытия видеофайла")
    exit()

# Получение параметров видео
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
duration = total_frames / fps

print(f"\nИнформация о видео:")
print(f"Размер: {frame_width}x{frame_height}")
print(f"Всего кадров: {total_frames}")
print(f"Длительность: {duration:.2f} секунд\n")

# Параметры выходного видео
output_path = 'output_video_sidewalk_segmented.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if show_window:
    # Создаем окно для отображения прогресса
    cv2.namedWindow('Sidewalk Segmentation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sidewalk Segmentation', 800, 600)

frame_count = 0
start_processing_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Сегментация кадра
    inference_start = time.time()
    result = inference_model(model, frame)
    inference_time = time.time() - inference_start
    
    seg_mask = result.pred_sem_seg.data[0].cpu().numpy()

    # Создаем маску для тротуара
    sidewalk_mask = (seg_mask == SIDEWALK_CLASS_ID).astype(np.uint8)
    
    # Применяем морфологические операции для улучшения маски
    kernel = np.ones((5, 5), np.uint8)
    sidewalk_mask = cv2.morphologyEx(sidewalk_mask, cv2.MORPH_CLOSE, kernel)
    
        # Создаем цветную маску (красный цвет для тротуара)
    color_mask = np.zeros_like(frame)
    # color_mask[sidewalk_mask == 1] = [0, 0, 255]  # Красный цвет
    color_mask[sidewalk_mask == 1] = [0, 255, 255]  # Жёлтый цвет (BGR)
    
    # Наложение маски на оригинальный кадр (без затемнения)
    # Вместо addWeighted просто добавляем цвет там, где есть маска
    blended = frame.copy()
    blended[sidewalk_mask == 1] = blended[sidewalk_mask == 1] * 0.5 + color_mask[sidewalk_mask == 1] * 0.5
    
    # Добавляем текст с информацией
    cv2.putText(blended, "Sidewalk Detection", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Сохранение кадра
    out.write(blended)
    
    # Расчет прогресса и скорости обработки
    frame_count += 1
    progress = (frame_count / total_frames) * 100
    elapsed_time = time.time() - start_processing_time
    remaining_time = (elapsed_time / frame_count) * (total_frames - frame_count)
    fps_processing = frame_count / elapsed_time
    
    if show_window:
        # Создаем изображение для отображения
        display_frame = blended.copy()
        
        # Добавляем информацию о прогрессе
        cv2.putText(display_frame, f"Progress: {progress:.1f}%", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (20, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Time: {elapsed_time:.1f}s | Remained: {remaining_time:.1f}s", (20, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Speed: {fps_processing:.2f} FPS", (20, 220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Рисуем прогресс-бар
        bar_width = int(frame_width * 0.9)
        bar_height = 30
        bar_x = (frame_width - bar_width) // 2
        bar_y = frame_height - 50
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + int(bar_width * progress / 100), bar_y + bar_height), (0, 255, 0), -1)
        
        # Отображаем кадр
        cv2.imshow('Sidewalk Segmentation', display_frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames}", end='\r')

# Освобождение ресурсов
cap.release()
out.release()
if show_window:
    cv2.destroyAllWindows()

total_time = time.time() - start_time
print("\nОбработка видео завершена!")
print(f"Общее время обработки: {total_time:.2f} секунд")
print(f"Средняя скорость обработки: {total_frames/total_time:.2f} FPS")
print(f"Результат сохранен в: {output_path}")