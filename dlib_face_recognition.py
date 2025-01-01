import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

def load_known_faces(faces_dir="face_photos"):
    """Load face encodings from the photos directory"""
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
        print(f"\nCreated {faces_dir} directory.")
        print("Please add your photos with format: name.jpg")
        return [], []
    
    known_face_encodings = []
    known_face_names = []
    
    print("\nLoading known faces...")
    for filename in os.listdir(faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # Get name from filename (without extension)
                name = os.path.splitext(filename)[0]
                
                # Load the image and get face encoding
                image_path = os.path.join(faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"Loaded face: {name}")
                else:
                    print(f"Warning: No face found in {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return known_face_encodings, known_face_names

def record_attendance(name, attendance_file):
    """Record attendance in CSV file"""
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H:%M:%S")
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Date', 'Time'])
    
    # Append attendance record
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])
    
    print(f"\nRecorded attendance for {name} at {time_str}")

def main():
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces()
    
    if not known_face_names:
        print("\nNo faces loaded. Please add photos to the face_photos directory.")
        print("Photo requirements:")
        print("1. Name the file as yourname.jpg (e.g., john.jpg)")
        print("2. Make sure your face is clearly visible")
        print("3. Only one face per image")
        return
    
    print(f"\nLoaded {len(known_face_names)} faces: {', '.join(known_face_names)}")
    
    # Initialize video capture
    print("\nInitializing camera...")
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set up attendance file
    attendance_file = f'attendance_{datetime.now().strftime("%Y-%m-%d")}.csv'
    recorded_names = set()  # Track names already recorded
    
    print("\nFace Recognition System Ready")
    print("Press 'q' to quit")
    
    while True:
        # Capture frame
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Find faces in frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Process each face in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Check if face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            confidence = 0
            
            if True in matches:
                # Find best match
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                
                if matches[best_match_index] and confidence > 0.5:
                    name = known_face_names[best_match_index]
                    
                    # Record attendance if not already recorded
                    if name not in recorded_names:
                        record_attendance(name, attendance_file)
                        recorded_names.add(name)
            
            # Draw box and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw box around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Add name and confidence
            label = f"{name} ({confidence:.1%})" if name != "Unknown" else name
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Face Recognition System', frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



