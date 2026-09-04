import cv2
import os
import re


def extract_frame_id(filename):
    match = re.search(r'Frame_(\d+)_Pred', filename)
    return int(match.group(1)) if match else -1


def extract_frame_id_noise(filename):
    match = re.search(r'frame_(\d+)_', filename)
    return int(match.group(1)) if match else -1


def frames_to_video(frame_dir, output_path, fps=30):

    # if not os.path.isdir(output_path):
    #     os.makedirs(output_path, exist_ok=True)

    # 获取所有图片文件，并排序（确保顺序正确）
    images = [img for img in os.listdir(frame_dir) if img.endswith((".png", ".jpg", ".jpeg"))]

    # ✅ 核心：按数字排序
    images = sorted(images, key=extract_frame_id_noise)

    if not images:
        print("Warning: No images found in the directory!")
        return

    # 读取第一张图片获取尺寸
    first_frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, _ = first_frame.shape

    # 定义编码器（mp4推荐使用mp4v）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(frame_dir, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Warning: skip {img_name}")
            continue

        video.write(frame)

    video.release()
    print(f"Video saved to {output_path}")


if __name__ == '__main__':

    frame_dir = '/data/rachel_data/leo/NExTVideo-derain/'

    for video_name in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, video_name)
        print(frame_path)

        frames_to_video(
        frame_dir=frame_path,          # 图片目录
        output_path='NExTVideo-denoise/{}'.format(video_name) + '.mp4',    # 输出视频
        fps=24                       # 帧率
        )