import cv2
import numpy as np 
from mmseg.apis import inference_model, init_model
from mmseg.utils import get_palette

VIDEO_PATH = "selenium_video_11s.mp4"
# Конфигурация модели и путь к весам
config_file = 'configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth'
# config_file = 'configs/deeplabv3plus/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.py'
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth'

# Инициализация модели
model = init_model(config_file, checkpoint_file, device='cpu')

# Палитра цветов для Cityscapes (19 классов)
palette = get_palette('cityscapes')

# Открытие видеофайла
cap = cv2.VideoCapture(VIDEO_PATH)

# Получение общего количества кадров для расчета прогресса
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Параметры выходного видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = 'output_video_segmented.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Сегментация кадра
    result = inference_model(model, frame)
    seg_mask = result.pred_sem_seg.data[0].cpu().numpy()

    # Преобразование маски в цветное изображение
    color_mask = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
    for class_id in np.unique(seg_mask):
        color_mask[seg_mask == class_id] = palette[class_id]

    # Наложение маски на оригинальный кадр
    alpha = 0.5
    blended = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

    # Сохранение кадра
    out.write(blended)
    
    # Расчет и вывод процента выполнения
    frame_count += 1
    progress = (frame_count / total_frames) * 100
    print(f"Прогресс обработки: {progress:.2f}%", end='\r')

# Освобождение ресурсов
cap.release()
out.release()
print("\nОбработка видео завершена.")