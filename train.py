import os
import random

import cv2
import numpy as np

from ultralytics import YOLO

dataset_path = "hmdb51"
alarming_actions = ['push', 'fall_floor', 'hit', 'kick',
                    'punch', 'run', 'shoot_gun', 'jump', 'throw']
normal_actions = ['stand', 'walk', 'sit']
actions = alarming_actions + normal_actions


def video_capture(video_path):
    print(f'Now processing - {video_path}')
    model = YOLO('yolov8x-pose')
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 30:
        return
    frame_interval = max(total_frames // 30, 1)
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_count >= 30:
            break

        if frame_count % frame_interval != 0:
            frame_count += 1
            continue
        frame_count += 1

        results = model.predict(frame)

        # annotated_frame = results[0].plot()
        # cv2.imshow(video_path, annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


for action in actions:
    action_dir = dataset_path+'/'+action+'/'
    file_names_all = os.listdir(action_dir)
    file_names_full_body = list(filter(lambda n: '_f_' in n, file_names_all))
    print(f'{action} - {len(file_names_full_body)}')
    file_names = random.sample(file_names_full_body, 1)
    print(f'Processing action - {action}')
    for file_name in file_names:
        video_capture(action_dir+file_name)
