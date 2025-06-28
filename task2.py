import cv2
from ultralytics import YOLO
import numpy as np
import time
import math

VIDEO_PATH = '15sec_input_720p.mp4'  
MODEL_PATH = 'best.pt'               
OUTPUT_VIDEO_PATH = 'tracked_players_output.mp4' 

TRACKER_CONFIG = 'botsort.yaml' 

# Maximum number of unique player IDs (1 to 24)
MAX_PLAYER_IDS = 24 
MAX_LOST_FRAMES_BUFFER = 60  # Approximately 2 seconds at 30 FPS

ultralytics_to_custom_id_map = {} 

# Keeps track of (1-24) are currently available
# We use a list and sort it to always assign the lowest available ID.
available_custom_ids = list(range(1, MAX_PLAYER_IDS + 1))
available_custom_ids.sort()

ultralytics_id_last_seen_frame = {}

# Keeps track of how many consecutive frames has been lost.
ultralytics_id_lost_counter = {}


def get_custom_player_id(ultralytics_track_id, current_frame_number):
    """
    Assigns or retrieves a custom player ID (1-24) for a given ultralytics_track_id.
    Manages the pool of available IDs and handles re-assignment for returning players.
    """
    ultralytics_id_last_seen_frame[ultralytics_track_id] = current_frame_number
    ultralytics_id_lost_counter[ultralytics_track_id] = 0

    # Check if ultralytics_track_id already has a custom ID
    if ultralytics_track_id in ultralytics_to_custom_id_map:
        return ultralytics_to_custom_id_map[ultralytics_track_id]
    else:
   # Assign a new custom ID.
        if available_custom_ids:

            new_custom_id = available_custom_ids.pop(0) 
            ultralytics_to_custom_id_map[ultralytics_track_id] = new_custom_id
            print(f"Assigned custom ID {new_custom_id} to new internal track ID {ultralytics_track_id}")
            return new_custom_id
        else:
            print(f"Warning: All {MAX_PLAYER_IDS} player IDs are in use! Cannot assign new ID.")
            return -1


def manage_lost_tracks(current_frame_number):
    """
    Identifies tracks that are no longer being detected and manages their custom IDs.
    If a track is lost beyond the buffer, its custom ID is returned to the pool.
    """
    global ultralytics_to_custom_id_map
    global available_custom_ids
    global ultralytics_id_lost_counter
    global ultralytics_id_last_seen_frame

    tracks_to_process = list(ultralytics_to_custom_id_map.keys()) 

    for ult_track_id in tracks_to_process:
        if ultralytics_id_last_seen_frame.get(ult_track_id, -1) < current_frame_number:
            ultralytics_id_lost_counter[ult_track_id] = ultralytics_id_lost_counter.get(ult_track_id, 0) + 1

            if ultralytics_id_lost_counter[ult_track_id] > MAX_LOST_FRAMES_BUFFER:
                custom_id = ultralytics_to_custom_id_map.pop(ult_track_id)
                available_custom_ids.append(custom_id)
                available_custom_ids.sort() # make it sorted
                
                ultralytics_id_lost_counter.pop(ult_track_id, None)
                ultralytics_id_last_seen_frame.pop(ult_track_id, None)

                print(f"Reclaimed custom ID {custom_id} from internal track ID {ult_track_id}. Now available IDs: {available_custom_ids}")

def process_video_and_track_players():
    """
    Main function to process the video, detect, track, and assign custom player IDs.
    """
    try:
        # Load the YOLO model
        model = YOLO(MODEL_PATH)
        print(f"YOLO model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames.")

    # Initialize video writer for output (optional)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"Output video will be saved to '{OUTPUT_VIDEO_PATH}'")

    frame_count = 0
    start_time = time.time()

    print(f"Starting tracking with '{TRACKER_CONFIG}' for players (IDs 1-{MAX_PLAYER_IDS})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model.track(source=frame, persist=True, tracker=TRACKER_CONFIG, conf=0.3, verbose=False)

        current_frame_detected_ultralytics_ids = set()

        if results and len(results) > 0 and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
            ultralytics_track_ids = results[0].boxes.id.int().cpu().numpy() # Internal tracker IDs
            scores = results[0].boxes.conf.cpu().numpy() # Confidence scores
            classes = results[0].boxes.cls.int().cpu().numpy() # Class IDs

            for i, (box, ult_track_id, score, class_id) in enumerate(zip(boxes, ultralytics_track_ids, scores, classes)):
                label = model.names[class_id] 

                x1, y1, x2, y2 = map(int, box)
                
                color = (0, 0, 0)
                text_color = (255, 255, 255)

                if label == 'player':
                    custom_player_id = get_custom_player_id(ult_track_id, frame_count)
                    
                    if custom_player_id != -1: 
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID: {custom_player_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        current_frame_detected_ultralytics_ids.add(ult_track_id)
                    else:
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 
                        cv2.putText(frame, "Player", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                elif label == 'ball':
                    color = (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, "Ball", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
   
                elif label == 'referee': 
                    color = (255, 0, 0) 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, "Referee", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for ult_track_id in list(ultralytics_to_custom_id_map.keys()):
            if ult_track_id not in current_frame_detected_ultralytics_ids:
                ultralytics_id_lost_counter[ult_track_id] = ultralytics_id_lost_counter.get(ult_track_id, 0) + 1
            else:
                ultralytics_id_lost_counter[ult_track_id] = 0

        manage_lost_tracks(frame_count)
        cv2.imshow('Player Re-Identification and Tracking', frame)
        out.write(frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Total processing time: {processing_time:.2f} seconds.")
    print(f"Processed {frame_count} frames at an average of {frame_count / processing_time:.2f} FPS.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed. Output saved.")

if __name__ == "__main__":
    process_video_and_track_players()

