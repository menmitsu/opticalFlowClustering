import os
import sys
import subprocess
from generateVectorsFromVideo import processVideo

def process_file(file_path,csv_file):
    
    cmd = ["python3", "generateVectorsFromVideo.py", "--video", file_path, "--csvFile", csv_file]
    subprocess.run(cmd, check=True)

def find_optical_mp4_files(folder,csv_file):
    mp4_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if '_optical.mp4' in file:
                file_path = os.path.join(root, file)
                mp4_files.append(file_path+"\n")
                processVideo(file_path,csv_file)
    return mp4_files

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python program.py <folder> csv_file")
        sys.exit(1)
    folder = sys.argv[1]
    csv_file=sys.argv[2]
    mp4_files = find_optical_mp4_files(folder,csv_file)
    # print(mp4_files)
