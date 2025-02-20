from deepface import DeepFace
import cv2
import os
import time
import threading
from datetime import datetime
import csv

camera_is_busy = False
current_frame = None
frame_lock = threading.Lock()
last_detected_person = None
total_attempts = successful_recognitions = 0
processing_times = []
confidence_scores = []

def calculate_averages():
    avg_rate = (successful_recognitions / total_attempts * 100) if total_attempts > 0 else 0
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    return avg_rate, avg_time, avg_conf
    if camera_is_busy:
        return
        
    camera_is_busy = True
    
    try:
        # Try to find faces using Deepfaces find function
        result = DeepFace.find(
            frame,
            db_path="face_photos",
            model_name=model,
            enforce_detection=False,
            detector_backend="mtcnn",
            distance_metric="cosine",
            silent=True
        )
        
        # If we found a face
        if len(result[0]['identity']) > 0:
            # Get person's name from the file path
            person = os.path.basename(result[0]['identity'][0]).split('.')[0]
            match_score = 1 - result[0]['distance'][0]
            
            # Get face position
            x = int(result[0]['source_x'][0])
            y = int(result[0]['source_y'][0])
            width = result[0]['source_w'][0]
            height = result[0]['source_h'][0]
            
            # Set min confidence based on model
            min_confidence = 0.65
            if model == "ArcFace":
                min_confidence = 0.45
            elif model == "Dlib":
                min_confidence = 0.85
            
            # Draw box and save to CSV if confidence is high enough
            if match_score >= min_confidence:
                # Green box for a match
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0,255,0), 2)
                
                # Save new person to CSV
                if person not in people_found:
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_file.writerow([time_now, person, f"{match_score:.2%}", model])
                    print(f"\nFound {person} at {time_now} ({match_score:.2%})")
                    people_found.add(person)
                    
                # Show name and the confidence
                text = f"{person} ({match_score:.1%})"
                cv2.putText(frame, text, (x+5, y-5), 
                          cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
            else:
                # Draw red box for unknown
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 2)
                cv2.putText(frame, "Unknown", (x+5, y-5), 
                          cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
        
        with frame_lock:
            current_frame = frame.copy()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        
    camera_is_busy = False

def main():
    # Choose a model out the ones we are evaluating
    models = ["ArcFace", "Facenet", "Dlib"]
    print("\nPick a model:")
    for i, m in enumerate(models, 1):
        print(f"{i}. {m}")
    model = models[int(input("\nEnter number (1-3): ")) - 1]
    
    # Make sure we have a photos folder
    if not os.path.exists("face_photos"):
        os.makedirs("face_photos")
        print("\nPlease put photos in face_photos folder")
        return
    
    # Setup camera and CSV file
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    csv_path = f'face_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Time', 'Name', 'Confidence', 'Model'])
    
    print("\nPress 'q' to quit or 'r' to reset")
    
    while True:
        # Get camera frame
        success, frame = camera.read()
        if not success:
            break
            
        # Check for faces every 300ms
        if time.time() % 0.3 < 0.1:
            threading.Thread(target=check_face, 
                           args=(frame.copy(), model, writer)).start()
        
        # Show frame
        display = current_frame if current_frame is not None else frame
        cv2.imshow('Face Recognition', display)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            people_found.clear()
    
    # Clean up
    csv_file.close()
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()