import cv2
import json
import os

# ===== 读取json =====
with open('outscores/nextqa/blip/nextqa-mc-test/video.json', 'r') as f:
    video_list = json.load(f)

with open('outscores/nextqa/blip/nextqa-mc-test/frames.json', 'r') as f:
    frames_list = json.load(f)

# ===== 输出目录 =====
out_dir = "output_videos-nextqa-mc-test"
os.makedirs(out_dir, exist_ok=True)

# ===== 主循环 =====
for vid_idx, (video_path, frame_ids) in enumerate(zip(video_list, frames_list)):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        continue

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频
    video_name = os.path.split(video_path)[-1]
    out_path = os.path.join(out_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"▶ 处理: {video_path}")

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ 读取失败 frame {fid}")
            continue
        writer.write(frame)

    cap.release()
    writer.release()

print("✅ 全部完成")