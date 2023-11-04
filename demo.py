import cv2
import os
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict


alarming_actions = ['hit', 'jump', 'punch', 'run', 'shoot_gun','throw']
normal_actions = ['walk']
processed_path = "hmdb51_processed"
actions = np.array(
    list(filter(lambda x: not x.startswith('.'), os.listdir(processed_path))))
label_map = {label: idx for idx, label in enumerate(actions)}

frames_map = defaultdict(lambda: deque(maxlen=16))

def save_frame(id, frame):
    if len(frames_map[id]) == 0:
        for i in range(16):
            frames_map[id].append(np.zeros_like(frame))
    frames_map[id].append(frame)

def get_frames(id):

    print(np.array([frames_map[id]]).shape)

    return np.array([frames_map[id]])

yolo = YOLO('yolov8n-pose.pt')
model = tf.keras.models.load_model('model.h5')
cap = cv2.VideoCapture("https://www.youtube.com/watch?v=9-gWNtrH1e0")

actual_fps = cap.get(cv2.CAP_PROP_FPS)
desired_fps = 10
skip_frame = actual_fps // desired_fps

frame_counter = 0
label = defaultdict(lambda: 'No Label')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1
    results = yolo.track(frame, persist=True)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    if len(boxes)>0:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
    else:
        ids = []
    print('ids', ids)
    i = 0
    for box, id in zip(boxes, ids):
        if frame_counter == skip_frame:
            keypoints = results[0].keypoints[i].xyn[0].numpy().flatten()
            save_frame(id, keypoints)
            frames = get_frames(id)
            prediction = model.predict(frames)
            print('Prediction ', prediction)
            label_idx = np.argmax(prediction)
            label[id] = actions[label_idx]

        if label[id] in alarming_actions:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (box[0], box[1]),
                      (box[2], box[3]), color, 2)
        
        cv2.putText(
            frame,
            f"Id {id} Lbl-{label[id]}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        i+=1

    frame_counter %= skip_frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
