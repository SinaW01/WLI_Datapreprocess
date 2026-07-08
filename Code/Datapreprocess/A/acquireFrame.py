#***********************     Split the video into frames    ******************************

import os
import cv2


def video_to_frames(
    video_path,
    output_root,
    image_ext=".jpg",
    target_fps=None
):


    if not os.path.isfile(video_path):
        raise ValueError(f"Video does not exist: {video_path}")

    if output_root is None:
        raise ValueError("output_root must be specified manually")


    video_name = os.path.splitext(os.path.basename(video_path))[0]

    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open the video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if target_fps is None:
        frame_interval = 1
        actual_fps = original_fps
    else:
        frame_interval = max(int(round(original_fps / target_fps)), 1)
        actual_fps = original_fps / frame_interval

    
    print("Start frame extraction:")
    print(f" Video path: {video_path}")
    print(f" Video name: {video_name}")
    print(f" Original FPS: {original_fps:.2f}")
    print(f" Target FPS: {target_fps if target_fps else 'All'}")
    print(f" Actual frame extraction FPS: {actual_fps:.2f}")
    print(f" Total frames: {total_frames}")
    print(f" Output directory: {output_dir}")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_name = f"{video_name}_frame_{saved_idx:05d}{image_ext}"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()

    print(f"Frame extraction completed successfully, {saved_idx} images saved in total")


if __name__ == "__main__":
    video_path = r"./inputIMG/20250507_095001.mp4"
    output_root = r"./outputIMG"

    video_to_frames(
        video_path=video_path,
        output_root=output_root,
        image_ext=".jpg",   
        target_fps=None    
    )
