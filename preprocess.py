import os

import cv2
import numpy as np

from ultralytics import YOLO

dataset_path = "hmdb51"
processed_path = "hmdb51_processed_filtered"
alarming_actions = ['hit', 'punch', 'run', 'shoot_gun']
normal_actions = ['walk']
actions = alarming_actions + normal_actions


def extract_keypoints(results):
    """
    Extracting keypoints value for each frames.
    """
    num_persons = len(results[0].boxes)
    print('Total persons:',num_persons)
    
    kp_flat = np.zeros(34)

    if num_persons > 0:
        kp_flat = results[0].keypoints[0].xyn[0].numpy().flatten()

    return kp_flat


def process_video(action, file_name, show=False, dry_run=False):
    """
    Process a video by taking 30 frames and extract keypoints from each frame.
    Then save the resultant numpy array in the processed directory for the LSTM network to train.
    """
    video_path = f'{dataset_path}/{action}/{file_name}'
    print(f'Processing - {video_path}')
    model = YOLO('yolov8x-pose')
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total frames -', total_frames)
    if total_frames < 16:
        print('Not enough frames. Aborting.')
        return
    frame_interval = max(total_frames // 16, 1)
    captured_frames = 0
    processed_frames = 0

    processed_np_array = np.empty((16, 34))
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if processed_frames >= 16:
            break

        if captured_frames % frame_interval != 0:
            captured_frames += 1
            continue

        captured_frames += 1

        print(f'Processing frame #{processed_frames}')
        results = model.predict(frame)
        keypoints = extract_keypoints(results)
        processed_np_array[processed_frames] = keypoints
        processed_frames += 1
        if show:
            annotated_frame = results[0].plot()
            cv2.imshow(video_path, annotated_frame)

    if not dry_run:
        processed_file_directory = os.path.join(processed_path, action)
        if not os.path.exists(processed_file_directory):
            os.mkdir(processed_file_directory)
        processed_file_path = f'{processed_file_directory}/{file_name[:-4]}.npy'
        np.save(processed_file_path, processed_np_array)
        print(f'Processed file saved at "{processed_file_path}"')

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


def preprocess_dataset():
    """
    Make frame by frame keypoint data from the dataset for the LSTM network to use.
    """
    for action in actions:
        print(f'Processing action - {action}')
        action_dir = dataset_path+'/'+action+'/'
        file_names_all = os.listdir(action_dir)
        file_names_full_body = list(
            filter(lambda n: n.startswith('_'), file_names_all))
        file_names = file_names_full_body
        print(f'Total files - {len(file_names)}')
        for file_name in file_names:
            process_video(action, file_name)


def run():
    preprocess_dataset()


if __name__ == '__main__':
    run()
