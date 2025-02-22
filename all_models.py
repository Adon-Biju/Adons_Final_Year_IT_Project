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
    get_historical_aggregate_stats,
    record_unsuccessful_recognition,
    get_unsuccessful_stats 
)

camera_is_busy = False
current_frame = None
frame_lock = threading.Lock()
last_detected_person = None
total_attempts = successful_recognitions = 0
processing_times = []
confidence_scores = []

def get_model_choice():
    while True:
        models = ["ArcFace", "Facenet", "Dlib"]
        print("\nPick a model:")
        for i, m in enumerate(models, 1):
            print(f"{i}. {m}")
        
        try:
            choice = input("\nEnter number (1-3): ").strip()
            if not choice:  
                print("Please enter a number between 1 and 3")
                continue
                
            number = int(choice)
            if 1 <= number <= 3:
                return models[number - 1]
            else:
                print("Please enter a number between 1 and 3")
        except ValueError:
            print("Please enter a valid number between 1 and 3")

def calculate_averages():
    avg_rate = (successful_recognitions / total_attempts * 100) if total_attempts > 0 else 0
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    return {
        'avg_rate': avg_rate,
        'avg_time': avg_time,
        'avg_confidence': avg_conf,
        'total_attempts': total_attempts,
        'successful_recognitions': successful_recognitions
    }

def check_face(frame, model):
    global camera_is_busy, current_frame, total_attempts, successful_recognitions, last_detected_person
    if camera_is_busy:
        return
    camera_is_busy = True
    
    try:
        cv2.putText(frame, "Please look directly at the screen", 
                   (int(frame.shape[1]/2) - 200, 30),  
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)
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

def display_statistics(stats, model_name):
    """Display current test results"""
    print(f"\nCurrent Test Results:")
    print("-" * 50)
    print(f"Model: {model_name}")
    print(f"Total Attempts: {stats['total_attempts']}")
    print(f"Successful Recognitions: {stats['successful_recognitions']}")
    print(f"Average Recognition Rate: {stats['avg_rate']:.1f}%")
    print(f"Average Processing Time: {stats['avg_time']:.3f} seconds")
    print(f"Average Confidence: {stats['avg_confidence']:.2%}")
    print("-" * 50)

def display_historical_stats(historical_stats):
    """Display historical statistics"""
    print("\nOverall Model Statistics:")
    print("-" * 50)
    for stat in historical_stats:
        print(f"Model: {stat['model_name']}")
        print(f"Total Tests: {stat['total_tests']}")
        print(f"Overall Recognition Rate: {stat['overall_recognition_rate']:.1f}%")
        print(f"Overall Processing Time: {stat['overall_processing_time']:.3f} seconds")
        print(f"Overall Confidence: {stat['overall_confidence']:.2%}")
        print("-" * 50)
    

def main():
    try:
     
        init_database()
        
    
        model = get_model_choice()
        
        if not os.path.exists("face_photos"):
            os.makedirs("face_photos")
            print("\nPlease put photos in face_photos folder")
            return
        
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
        
            stats = calculate_averages()
            if save_test_results(model, last_detected_person, stats):
                print("\nResults saved to database successfully!")
            else:
                print("\nFailed to save results to database.")
            
           
            display_statistics(stats, model)
            
           
            if save_aggregate_stats():
                print("\nAggregate statistics saved successfully!")
                historical_stats = get_historical_aggregate_stats()
                display_historical_stats(historical_stats)
        else:
            print("\n---------No known faces were recognized------------")

            record_unsuccessful_recognition(model)
            print("\nUnsuccessful Recognition Statistics:")
            print("-" * 50)
            unsuccessful_stats = get_unsuccessful_stats()
            for stat in unsuccessful_stats:
                print(f"Model: {stat['model_name']}")
                print(f"Unsuccessful Attempts: {stat['unsuccessful_count']}")
                if stat['last_updated']:
                    print(f"Last Failed: {stat['last_updated']}")
                print("-" * 50)
            
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()