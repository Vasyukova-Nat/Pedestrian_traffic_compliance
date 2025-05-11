import numpy as np

# VIDEO_PATH = "video03.mp4"
VIDEO_PATH = "selenium_video_14.mp4"


''' Детекция '''
YOLO_MODEL_PATH = "yolov8m.pt" 
OUTPUT_DETECTION_FILE = "output.mp4"

# Версии:
# "yolov8n.pt" - nano (самая легкая)
# "yolov8s.pt" - small
# "yolov8m.pt" - medium
# "yolov8l.pt" - large
# "yolov8x.pt" - xlarge (самая точная)


''' Сегментация '''
SEGMENTATION_MODEL_PATH = "yolov8x-seg.pt"  # Сегментация
# Версии:
# yolov8n-seg.pt 
# yolov8s-seg.pt
# yolov8m-seg.pt 
# yolov8l-seg.pt 
# yolov8x-seg.pt


''' Захват видео с камеры '''
# URL = "https://flussonic2.powernet.com.ru:444/user102078/embed.html?token=dont-panic-and-carry-a-towel&autoplay=true&play_duration=28800"
# URL = "https://flussonic2.powernet.com.ru:444/user71382/embed.html?token=dont-panic-and-carry-a-towel&amp;autoplay=true&amp;play_duration=28800"
# URL = "https://flussonic2.powernet.com.ru:444/user55644/embed.html?token=dont-panic-and-carry-a-towel&amp;autoplay=true&amp;play_duration=28800"
URL = "https://flussonic2.powernet.com.ru:444/user55644/embed.html?token=dont-panic-and-carry-a-towel&autoplay=true&play_duration=28800"

OUTPUT_FILE = "selenium_video.mp4"


''' Разметка пешеходного перехода '''
crosswalk_coordinates = np.array([ 
    [731, 384],
    [610, 395],
    [610, 563],
    [800, 563]
], dtype=np.int32) # коорд-ты в порядке обхода