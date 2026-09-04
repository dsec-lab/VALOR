import os
import shutil
import albumentations as A
import cv2
from PIL import Image
import argparse
import numpy as np
import torch
from torchvision.models import resnet50
import torchvision.transforms as T
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, AdversarialPatchPyTorch
from art.attacks.evasion import CarliniL0Method, ProjectedGradientDescentPyTorch


def load_video(video_path):
    # Read a video with OpenCV and convert it to the RGB colorspace
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    print('number of frames: {}'.format(len(frames)))
    cap.release()

    return np.array(frames)


def save_video(aug_frames, aug_video_path):
    # Read a video with OpenCV and convert it to the RGB colorspace
    h, w, _ = aug_frames[0].shape
    out = cv2.VideoWriter(
        aug_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        24,
        (w, h)
    )
    for f in aug_frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

    return


def build_transform(args):
    # Declare an augmentation pipeline
    transform = None
    if args.brightness:
        # transform = A.RandomBrightnessContrast(
        #     brightness_limit=[-0.8, -0.6],
        #     brightness_by_max=False,
        #     contrast_limit=[-0.2, 0],
        #     p=1.0)
        
        transform = A.RandomBrightnessContrast(
            brightness_limit=[-0.01, 0.01],
            brightness_by_max=False,
            contrast_limit=[-0.01, 0.01],
            p=0.1
        )

    if args.blur:
        transform = A.GaussianBlur(
            sigma_limit=(3.0, 7.0),
            blur_limit=0,
            p=1.0)
    if args.motionblur:
        # https://albumentations.ai/docs/api-reference/albumentations/augmentations/blur/transforms/#MotionBlur
        # transform = A.MotionBlur(blur_limit=(10, 12),     # Strong blur
        #     angle_range=(-5, 5),     # Near-horizontal motion (±5°)
        #     direction_range=(0, 0),  # Symmetric blur (equally in both directions)
        #     p=1.0                    # Always apply
        #     )
        transform = A.MotionBlur(
            blur_limit=(9, 12),  # Variable strength
            angle_range=(0, 360),  # Any angle
            direction_range=(-1.0, 1.0),  # Any direction bias
            allow_shifted=True,  # Allow kernel to be shifted from center
            p=1.0)
    if args.downscale:
        transform = A.Downscale(
            scale_range=(0.25, 0.5),
            # interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR},
            interpolation_pair={'downscale': 0, 'upscale': 0},
            p=1.0)
    if args.pepper:
        transform = A.SaltAndPepper(
            amount=(0.1, 0.2),  # 10-20% of pixels will be noisy
            salt_vs_pepper=(0.7, 0.9),  # 70-90% of noise will be salt
            p=1.0)
    if args.noise:
        # Alternatives methods: AdditiveNoise
        transform = A.GaussNoise(std_range=(0.1, 0.2), p=1.0)  # 10-20% of max value
    if args.occlusion:
        transform = A.CoarseDropout(
            num_holes_range=(3, 6),
            hole_height_range=(0.1, 0.2),
            hole_width_range=(0.1, 0.2),
            fill="random_uniform",
            p=1.0)
    if args.fog:
        # slow
        # transform = A.Compose([
        #     A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.2), contrast_limit=(-0.2, 0.1), p=1.0),
        #     A.RandomFog(fog_coef_range=(0.05, 0.07), alpha_coef=0.1, p=1.0)
        # ])

        transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, -0.1),   # 中度变暗
                contrast_limit=(-0.1, 0.05),      # 轻微对比度变化
                p=1.0
            ),
            A.RandomFog(
                fog_coef_range=(0.05, 0.07),      # 中等雾化
                alpha_coef=0.1,
                p=1.0
            )
        ])

    if args.rain:
        transform = A.RandomRain(
            slant_range=(-15, 15),
            drop_length=30,
            drop_width=2,
            drop_color=(180, 180, 180),
            blur_value=9,
            brightness_coefficient=0.5,
            rain_type="heavy",
            p=1.0)
        
        # transform = A.RandomRain(
        #     slant_range=(0, 0),              # 不倾斜
        #     drop_length=5,                   # 很短的雨滴
        #     drop_width=1,                    # 最细
        #     drop_color=(200, 200, 200),      # 更浅
        #     blur_value=1,                    # 几乎不模糊
        #     brightness_coefficient=0.95,     # 几乎不变暗
        #     rain_type="drizzle",             # 最轻雨
        #     p=0.01                            # 10%概率触发
        # )
    if args.snow:
        # transform = A.RandomSnow(
        #     snow_point_range=(0.2, 0.4),
        #     brightness_coeff=2.0,
        #     method="texture",
        #     p=1.0)
        
        transform = A.RandomSnow(
            snow_point_range=(0.2, 0.4),
            brightness_coeff=2.0,
            method="texture",
            p=0.05)

    if args.spatter:
        transform = A.Spatter(
            mode="rain",
            mean=(0.65, 0.65),  # Higher mean = more coverage
            std=(0.3, 0.3),  # Lower std = more uniform effect
            cutout_threshold=(0.68, 0.68),  # Lower threshold = more drops
            intensity=(0.6, 0.6),  # Higher intensity = more visible effect
            color=(238, 238, 175),  # Blueish rain drops
            p=1.0)

        # transform = A.Spatter(
        #     mode="mud",
        #     mean=(0.55, 0.55),
        #     std=(0.25, 0.25),
        #     cutout_threshold=(0.7, 0.7),
        #     intensity=(0.6, 0.6),
        #     color=(120, 40, 40),  # Reddish-brown mud
        #     p=1.0)
    # if args.mosaic:
    #     transform = A.Mosaic(
    #         grid_yx=(2, 2),
    #         target_size=(200, 200),
    #         cell_shape=(120, 120),
    #         center_range=(0.4, 0.6),
    #         fit_mode="cover",
    #         p=1.0)
    if args.mask:
        transform = A.XYMasking(
            num_masks_x=[1, 3],
            num_masks_y=[1, 3],
            mask_x_length=[30, 50],
            mask_y_length=[30, 50],
            fill=0,
            p=1.0
        )
    pipeline = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.9], brightness_by_max=False, contrast_limit=[-0.2, 0.2],
                                   p=1.0),
        A.GaussianBlur(sigma_limit=(3.0, 7.0), blur_limit=0, p=1.0),  # 30% chance of applying
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3)
    ])
    return transform


def build_adversarial(frames, output_video_path):
    # 1. Load the pre-trained model
    model = resnet50(pretrained=True)
    model.eval()

    # 2. Packaged as an ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=None,  # 不训练
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0.0, 1.0)
    )

    # 3. FGSM attack
    fgsm = FastGradientMethod(classifier, eps=0.03)
    fgsm_frames = []
    for idx, frame in enumerate(frames):
        if idx % 100 == 0: print(idx)
        transform = T.Compose([
            # T.Resize((224, 224)),
            T.ToTensor()  # → [0,1], CHW
        ])
        x = transform(frame).numpy()[None, ...]  # (1,3,H,W)
        x_adv = fgsm.generate(x=x)
        fgsm_frames.append((x_adv * 255).round().clip(0, 255).astype("uint8"))
    fgsm_frames = np.transpose(np.concatenate(fgsm_frames, axis=0), (0, 2, 3, 1))
    save_video(fgsm_frames, output_video_path)


def get_args():
    parser = argparse.ArgumentParser(description="Choose exactly one augmentation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--brightness", action="store_true", help="use brightness augmentation")
    group.add_argument("--blur", action="store_true", help="use blur augmentation")
    group.add_argument("--motionblur", action="store_true", help="use motionblur augmentation")
    group.add_argument("--downscale", action="store_true", help="use downscale augmentation")
    group.add_argument("--pepper", action="store_true", help="use pepper augmentation")
    group.add_argument("--noise", action="store_true", help="use noise augmentation")
    group.add_argument("--occlusion", action="store_true", help="use occlusion augmentation")
    group.add_argument("--fog", action="store_true", help="use fog augmentation")
    group.add_argument("--rain", action="store_true", help="use rain augmentation")
    group.add_argument("--spatter", action="store_true", help="use spatter augmentation")
    group.add_argument("--snow", action="store_true", help="use snow augmentation")
    group.add_argument("--mosaic", action="store_true", help="use mosaic augmentation")
    group.add_argument("--mask", action="store_true", help="use mask augmentation")
    group.add_argument("--fgsm", action="store_true", help="use fgsm augmentation")

    parser.add_argument("--output_dir", type=str, default=None, help="output video type directory")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    input_data_dir = '/home/rachel/.cache/huggingface/mvbench_video-normal/Moments_in_Time_Raw/videos/validation/wetting/'
    output_data_dir_root = '/home/rachel/projects/leo/restriected_videos/Moments_in_Time_Raw'
    if not os.path.exists(output_data_dir_root):
        os.makedirs(output_data_dir_root)

    transform_type = args.output_dir
    output_data_dir = os.path.join(output_data_dir_root, transform_type)

    for root, dirs, files in os.walk(input_data_dir):
        for file in files:
            input_file_path = os.path.join(root, file)
            new_root = root.replace(input_data_dir, output_data_dir)
            if os.path.isdir(new_root):
                pass
            else:
                os.makedirs(new_root)

            if input_file_path.endswith(".mp4"):
                if file in os.listdir(new_root):
                    print('file existed~')
                    continue
                # load video as frames
                frames = load_video(input_file_path)
                output_video_path = os.path.join(new_root, file)

                if args.fgsm:
                    build_adversarial(frames, output_video_path)
                else:
                    transform = build_transform(args)
                    # print(transform)
                    # Augment a video
                    transformed = transform(images=frames)
                    aug_frames = transformed["images"]
                    # Save an Augmented video
                    if args.snow or args.brightness or args.noise or args.pepper:
                        save_video(aug_frames, aug_video_path='temp_snows.mp4')
                        cmd = "ffmpeg -y -i {0} -b:v 1M {1}".format('temp_snows.mp4', output_video_path)
                        os.system(cmd)
                        os.remove('temp_snows.mp4')
                    elif args.blur or args.rain or args.downscale or args.fog:
                        # fog so slow
                        print('saving video: {}'.format(output_video_path))
                        save_video(aug_frames, aug_video_path=output_video_path)
            else:
                shutil.copyfile(input_file_path, os.path.join(new_root, file))