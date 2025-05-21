import cv2
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from config import URL, OUTPUT_FILE


DURATION_SECONDS = 40
FRAME_RATE = 10  # Частота кадров. Классно на 10.

chrome_options = Options() # Настройки Selenium
chrome_options.add_argument("--headless")  # Фоновый режим
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1280,720")  # вместо 1920,1080

driver = webdriver.Chrome(options=chrome_options)
driver.get(URL)

try:
    WebDriverWait(driver, 40).until( # Ждем загрузки видео
        EC.presence_of_element_located((By.TAG_NAME, "video"))
    )
except Exception as e:
    print("Не удалось найти видео элемент:", e)
    driver.quit()
    exit()

video_element = driver.find_element(By.TAG_NAME, "video")
width = video_element.size['width']
height = video_element.size['height']

# Настройка VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'avc1') # Кодек. Можно: avc1, mp4v, X264
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FRAME_RATE, (width, height), isColor=True)

start_time = time.time()
end_time = start_time + DURATION_SECONDS
print(f"Захват видео на {DURATION_SECONDS} секунд...")
try:
    while time.time() < end_time:
        try:
            screenshot = driver.get_screenshot_as_png() # скриншот страницы
            
            # конвертируем в массив numpy
            img_arr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            # video_img = img
            x, y = video_element.location['x'], video_element.location['y']
            video_img = img[y:y+height, x:x+width]
            
            # Изменяем размер, если нужно (на случай расхождений)
            if video_img.shape[1] != width or video_img.shape[0] != height:
                video_img = cv2.resize(video_img, (width, height))
            
            # Конвертируем в правильный цвет (BGR для OpenCV)
            video_img = cv2.cvtColor(video_img, cv2.COLOR_RGB2BGR)
            
            # записываем кадр
            out.write(video_img)
            
            # задержка для поддержания FPS
            elapsed = time.time() - start_time
            expected_frames = int(elapsed * FRAME_RATE)
            actual_frames = out.get(cv2.CAP_PROP_FRAME_COUNT)
            if actual_frames < expected_frames:
                time.sleep(max(0, (1/FRAME_RATE) - (time.time() - start_time - actual_frames/FRAME_RATE)))
            
        except Exception as e:
            print("Ошибка при захвате кадра:", e)
            break

finally:
    out.release()
    driver.quit()
    
    if cv2.VideoCapture(OUTPUT_FILE).get(cv2.CAP_PROP_FRAME_COUNT) > 0:
        print(f"Видео успешно сохранено в {OUTPUT_FILE}")
    else:
        print("Не удалось сохранить видео. Файл может быть поврежден.")