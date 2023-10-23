import cv2
import os

def save_frames_from_video(video_path, output_folder, time_interval):

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) * time_interval) == 0:
            # Save the frame as a JPEG file
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()

if __name__ == "__main__":
    video_path = "videos/5_meio.h264"  # Replace with your video file path
    output_folder = "5_meio"  # Output folder for frames

    save_frames_from_video(video_path, output_folder, time_interval=0.1)
