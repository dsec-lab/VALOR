import cv2
import os
import shutil

"""
This script converts a video into individual frames and saves them to a specified directory.
Only a subset of frames is saved based on a defined interval to match the desired frame rate for further processing.
"""


# Path to the video file
# Directory to save the frames
video_root_dir = '/home/rachel/projects/leo/restriected_videos/NExTVideo/rain'
for root, dirs, files in os.walk(video_root_dir):
    for file in files:
        if file.endswith('.mp4'):
            video_path = os.path.join(root, file)
            output_folder = os.path.join('NExTVideo_rain', file.split('.')[0])
            print(output_folder)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            # Load the video
            cap = cv2.VideoCapture(video_path)

            # Get the frame rate of the video
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames_per_second = 10
            interval = int(fps / frames_per_second)

            frame_count = 0
            saved_frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame if it's in the interval
                if frame_count % interval == 0:
                    frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count:04d}.jpg')
                    cv2.imwrite(frame_filename, frame)
                    saved_frame_count += 1

                frame_count += 1

            cap.release()
            print(f'Extracted {saved_frame_count} frames to {output_folder}')