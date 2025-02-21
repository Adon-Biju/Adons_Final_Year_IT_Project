from deepface import DeepFace
import cv2
import os
import time
import threading
from datetime import datetime

from database_operations import (
    init_database, 
    save_test_results, 
    get_model_stats,
    save_aggregate_stats,
    get_historical_aggregate_stats
)

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

def check_face(frame, model):
    global camera_is_busy, current_frame, total_attempts, successful_recognitions, last_detected_person
    if camera_is_busy:
        return
    camera_is_busy = True
    
    try:
        start_time = time.time()
        face_objs = DeepFace.extract_faces(frame, detector_backend="mtcnn", enforce_detection=False)
        
        if len(face_objs) > 0:
            total_attempts += 1
            facial_area = face_objs[0]['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            result = DeepFace.find(frame, db_path="face_photos", model_name=model,
                                 enforce_detection=False, detector_backend="mtcnn",
                                 distance_metric="cosine", silent=True)
            
            processing_times.append(time.time() - start_time)
            
            if len(result[0]['identity'].values) > 0:
                person = os.path.basename(result[0]['identity'].values[0]).split('.')[0]
                match_score = 1 - result[0]['distance'].values[0]
                min_confidence = 0.45 if model == "ArcFace" else 0.85 if model == "Dlib" else 0.65
                
                if match_score >= min_confidence:
                    successful_recognitions += 1
                    confidence_scores.append(match_score)
                    last_detected_person = person
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{person} ({match_score:.1%})", (x+5, y-5), 
                              cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
                    print(f"Recognized {person} with confidence: {match_score:.1%}")
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
                    cv2.putText(frame, "Unknown", (x+5, y-5), 
                              cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
                cv2.putText(frame, "Unknown", (x+5, y-5), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
        
        with frame_lock:
            current_frame = frame.copy()
    except Exception as e:
        print(f"Error: {str(e)}")
    
    camera_is_busy = False

def main():
    models = ["ArcFace", "Facenet", "Dlib"]
    print("\nPick a model:")
    for i, m in enumerate(models, 1):
        print(f"{i}. {m}")
    model = models[int(input("\nEnter number (1-3): ")) - 1]
    
    if not os.path.exists("face_photos"):
        os.makedirs("face_photos")
        print("\nPlease put photos in face_photos folder")
        return
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nRunning face recognition for 15 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 15:
        success, frame = camera.read()
        if not success:
            break
        
        if time.time() % 0.3 < 0.1:
            threading.Thread(target=check_face, args=(frame.copy(), model)).start()
        
        cv2.imshow('Face Recognition', current_frame if current_frame is not None else frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if total_attempts > 0 and last_detected_person and successful_recognitions > 0:
        csv_path = f'face_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Time', 'Name', 'Model', 'Average_Confidence',
                           'Average_Recognition_Time_Seconds', 'Average_Recognition_Rate_Percent'])
            
            avg_rate, avg_time, avg_conf = calculate_averages()
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                last_detected_person, model,
                f"{avg_conf:.2%}", f"{avg_time:.3f}", f"{avg_rate:.1f}%"
            ])
        
        print(f"\nModel: {model}")
        print(f"Total Attempts: {total_attempts}")
        print(f"Successful Recognitions: {successful_recognitions}")
        print(f"Average Recognition Rate: {avg_rate:.1f}%")
        print(f"Average Processing Time: {avg_time:.3f} seconds")
        print(f"Average Confidence: {avg_conf:.2%}")
    else:
        print("\nNo known faces were recognized")
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()