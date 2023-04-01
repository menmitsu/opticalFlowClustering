import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def trim_video(video_path, output_path):
    """Trims a video into 3-second segments and saves them to the output directory."""
    video = VideoFileClip(video_path)
    duration = video.duration
    if duration > 3:
        for i in range(0, int(duration), 3):
            start = i
            end = min(i+3, duration)
            segment = video.subclip(start, end)
            
            # Save the trimmed segment to the output directory with the same name as the original file
            segment_path = os.path.join(output_path, f"{os.path.basename(video_path)}_{start}-{end}.mp4")
            segment.write_videofile(segment_path, codec="libx264")
            
    video.close()

# Get the input directory path from the command-line argument
import sys
if len(sys.argv) < 2:
    print("Usage: python trim_videos.py <input_directory>")
    sys.exit(1)
input_dir = sys.argv[1]

# Get the absolute path of the input directory
input_dir = os.path.abspath(input_dir)

# Create a new directory called "trimmed" in the parent directory
parent_dir = os.path.dirname(input_dir)
output_dir = os.path.join(parent_dir, "trimmed")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Recursively go through all directories and files in the input directory
for root, dirs, files in os.walk(input_dir):
    # Create the corresponding subdirectories in the output directory
    rel_root = os.path.relpath(root, input_dir)
    output_root = os.path.join(output_dir, rel_root)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Trim all .mp4 files in the current directory
    for file in files:
        if file.endswith(".mp4"):
            # Get the absolute path of the input file
            input_file = os.path.join(root, file)
            
            # Trim the video and save it to the corresponding subdirectory in the output directory
            trim_video(input_file, output_root)
