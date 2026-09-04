import cv2
import numpy as np


def create_video_strip(video_path, output_path, num_frames=5, padding=5):
    """
    将视频等间隔抽取指定数量的帧，并拼接成带白色分隔线的水平长图
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算等间隔采样的步长
    step = max(1, total_frames // num_frames)

    frames = []
    for i in range(num_frames):
        # 定位到指定的帧
        frame_idx = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # 如果不是最后一帧，在右侧添加白色 padding 作为分割线
            if i < num_frames - 1 and padding > 0:
                frame = cv2.copyMakeBorder(
                    frame, 0, 0, 0, padding,
                    cv2.BORDER_CONSTANT, value=[255, 255, 255]
                )
            frames.append(frame)

    cap.release()

    if frames:
        # 水平拼接所有帧
        final_strip = np.hstack(frames)
        cv2.imwrite(output_path, final_strip)
        print(f"✅ 长图已保存至: {output_path}")
    else:
        print("❌ 无法读取视频帧")


# 使用示例：从 video.mp4 中等间隔抽 6 帧，帧之间留 5 像素白边
create_video_strip('./Moments_in_Time_Raw/fog/getty-sprinkler-video-id453401937_2.mp4', 
                   './Moments_in_Time_Raw/fog/fog.png', num_frames=8, padding=5)