# Task - 2:⚽ Re-identification in a Single Feed

This project provides a robust solution for detecting and re-identifying football players within a video feed, ensuring consistent ID assignment even when players leave and re-enter the frame. It leverages YOLO for object detection and BoT-SORT for multi-object tracking with re-identification capabilities, topped with a custom ID management system.

# Features:
Player & Ball Detection: Utilizes a custom-trained YOLO model (best.pt) to accurately detect players and the ball.

Robust Multi-Object Tracking: Integrates BoT-SORT, an advanced tracking algorithm, for smooth and persistent tracking of objects across frames.

Player Re-Identification: Maintains unique player IDs even when players are temporarily occluded or go out of frame and reappear.

Fixed ID Pool (1-24): Assigns player IDs from a controlled pool of 1 to 24, ensuring IDs are reused efficiently.

Referee Detection Support: Includes visualization support for a 'referee' class, assuming the YOLO model is trained to detect it.

Video Output: Generates an output video with detected objects and assigned player IDs.

# Tech Stack
Language: Python 3.10

Core Libraries:

ultralytics: For YOLO model inference and integrated object tracking (BoT-SORT/ByteTrack).

opencv-python: For video processing (reading frames, drawing, writing output video).

numpy: For numerical operations (a dependency of the above).

Object Detection Model: Custom best.pt (YOLOv11 fine-tuned for players and ball).

# How to Set Up and Run the Code
Follow these steps to get the project up and running on your local machine.

1. Prerequisites
Before you begin, ensure you have the following installed:

Python 3.x: (e.g., Python 3.8 or newer)

pip: Python package installer (usually comes with Python).

2. Project Files
Make sure you have the following files in your project directory:

task2.py (The main Python script provided)

15sec_input_720p.mp4 (Your input video file)

best.pt (Your custom YOLOv11 object detection model)

botsort.yaml (The configuration file for the BoT-SORT tracker)

Important: This botsort.yaml file is part of the ultralytics library. You can find it in your Python environment's site-packages, e.g., /path/to/your/python/env/lib/pythonX.Y/site-packages/ultralytics/cfg/trackers/botsort.yaml. Copy this file into the same directory as task2.py.

3. Environment Setup
It's highly recommended to use a virtual environment to manage dependencies.

# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install required Python packages
pip install -r requirements.txt

requirements.txt content:
opencv-python>=4.5.0
ultralytics>=8.0.0
torch>=1.8.0 # Adjust based on your system (CPU or CUDA)
numpy>=1.20.0

Note on torch: If you have an NVIDIA GPU, installing a CUDA-enabled version of torch will significantly speed up model inference. Refer to the PyTorch website for specific installation commands for your CUDA version. For CPU-only, use:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

4. Running the Code
Once the environment is set up and files are in place, run the main script:

python task2.py

The script will:

Load the YOLO model and the video.

Process the video frame by frame, performing detection and tracking.

Display the tracked players with their assigned IDs (1-24) and the ball.

Save an output video named tracked_players_output.mp4 in the same directory.

Print progress and performance statistics to the console.

Press q on the display window to stop the video processing prematurely.

# Configuration
You can adjust key parameters directly within the task2.py file:

# --- Configuration ---
VIDEO_PATH = '15sec_input_720p.mp4'  # Path to your input video
MODEL_PATH = 'best.pt'               # Path to your YOLO model
OUTPUT_VIDEO_PATH = 'tracked_players_output.mp4' # Output video path (optional)

TRACKER_CONFIG = 'botsort.yaml'      # Tracker config: 'botsort.yaml' (recommended) or 'bytetrack.yaml'

MAX_PLAYER_IDS = 24                  # Maximum number of unique player IDs (1 to 24)
MAX_LOST_FRAMES_BUFFER = 60          # Frames a track can be 'lost' before reclaiming its custom ID
                                     # (adjust based on video FPS and expected occlusion duration)

For advanced tuning of the tracking behavior (e.g., to reduce ID switches), you can directly edit the botsort.yaml file you copied:

max_age: Increase this (e.g., from 30 to 90 or 120) to allow the tracker to retain a track for longer periods if a player is briefly out of sight or occluded. This is crucial for re-identification stability.

track_high_thresh / track_low_thresh: Adjust detection confidence thresholds used by the tracker.

max_iou_distance / max_cosine_distance: Fine-tune the matching criteria based on spatial overlap and appearance similarity.

# Notes on Referee Detection
Your provided best.pt model might not include a 'referee' class. If referees are not being detected, you will need to:

Collect Data: Gather images/frames containing referees.

Annotate Data: Label the referees with bounding boxes and a 'referee' class.

Retrain/Fine-tune YOLO Model: Use ultralytics to fine-tune your best.pt model (or train a new one) on this expanded dataset. Update MODEL_PATH to point to your newly trained model.

The current football_tracker_reid.py code includes visualization logic for a 'referee' class, ready for when your model supports it.

Thank you.
