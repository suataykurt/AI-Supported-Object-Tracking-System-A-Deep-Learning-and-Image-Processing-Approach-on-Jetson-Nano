import cv2
import torch
import numpy as np
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import time
import warnings
from typing import Tuple, Optional

import board
import busio
from adafruit_pca9685 import PCA9685

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 60  

PAN_SERVO_MIN = 1000  
PAN_SERVO_MAX = 5000  
TILT_SERVO_MIN = 1000
TILT_SERVO_MAX = 5000

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('E:/test/yolov5', 'custom', path='E:/test/yolov5s_custom/weights/best.pt', source='local')
model.to(device)
model.conf = 0.35
model.iou = 0.35
model.eval()
torch.backends.cudnn.benchmark = True

kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
kamera.set(cv2.CAP_PROP_FPS, 30)
kamera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
kamera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

prev_frame_time = 0
new_frame_time = 0
fps_array = []
frame_counter = 0
last_processed_frame = None

tracking_enabled = True
tracking_threshold = 0.5
frame_center = (320, 240)
tracking_box = None

def get_object_center(box: np.ndarray) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def calculate_servo_angles(center: Tuple[int, int]) -> Tuple[float, float]:
    pan_angle = (center[0] / 640) * (PAN_SERVO_MAX - PAN_SERVO_MIN) + PAN_SERVO_MIN
    tilt_angle = (center[1] / 480) * (TILT_SERVO_MAX - TILT_SERVO_MIN) + TILT_SERVO_MIN
    return pan_angle, tilt_angle

def set_servo_angle(pan_angle: float, tilt_angle: float):
    pca.channels[0].duty_cycle = int(pan_angle)
    pca.channels[1].duty_cycle = int(tilt_angle)

def process_frame(frame: np.ndarray) -> np.ndarray:
    global tracking_box

    with torch.no_grad():
        results = model(frame)
        
    frame_with_boxes = frame.copy()
    detections = results.xyxy[0].cpu().numpy()
    
    if len(detections) > 0:
        best_detection = detections[np.argmax(detections[:, 4])]
        if best_detection[4] >= model.conf:
            x1, y1, x2, y2, conf, cls = best_detection
            cv2.rectangle(frame_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            label = f'{model.names[int(cls)]} %{int(conf * 100)}'
            cv2.putText(frame_with_boxes, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            center = get_object_center(best_detection[:4])
            
            cv2.circle(frame_with_boxes, frame_center, 5, (0, 0, 255), -1)
            cv2.circle(frame_with_boxes, center, 5, (255, 0, 0), -1)
            
            if tracking_enabled:
                pan_angle, tilt_angle = calculate_servo_angles(center)
                set_servo_angle(pan_angle, tilt_angle)

            tracking_box = (x1, y1, x2, y2)
    
    return frame_with_boxes

while True:
    ret, frame = kamera.read()
    if not ret:
        break

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    
    fps_array.append(fps)
    if len(fps_array) > 30:
        fps_array.pop(0)
    avg_fps = sum(fps_array) / len(fps_array)

    if frame_counter % 2 == 0:
        processed_frame = process_frame(frame)
        last_processed_frame = processed_frame
    else:
        processed_frame = last_processed_frame if last_processed_frame is not None else frame

    frame_counter += 1

    cv2.putText(processed_frame, f'FPS: {int(avg_fps)}', (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(processed_frame, f'Takip: {"Acik" if tracking_enabled else "Kapali"}', (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Nesne Takip Sistemi", processed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q')]:
        break
    elif key == ord('t'):
        tracking_enabled = not tracking_enabled

kamera.release()
cv2.destroyAllWindows()
