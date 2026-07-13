import cv2
import time
from ultralytics import YOLO

print("Loading AI Model...")
model = YOLO('yolov8n.pt')
print("AI Model Loaded!")

cap = cv2.VideoCapture(1) 
cheating_objects = ['cell phone', 'laptop', 'book', 'tv', 'remote']
score = 100
flagged_events = []

face_missing_timer = 0

print("\n[AI SCAN STARTED] Camera ON")
print("Press 'q' to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    
    person_detected = False
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            
            if label == 'person':
                person_detected = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, f"person", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                face_missing_timer = 0 # reset timer if person is there
            
            if label in cheating_objects and conf > 0.5:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                cv2.putText(frame, f"{label} ALERT!", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                score -= 40
                flagged_events.append(f"{label} detected")
                print(f"ALERT: {label} detected! -40 points")
    
    # NEW LOGIC: If person disappears for 2 seconds = looking away/down
    if not person_detected:
        face_missing_timer += 1
        if face_missing_timer > 20: # ~2 seconds
            cv2.putText(frame, "WARNING: Face Not Visible", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if face_missing_timer == 21: # only deduct once
                score -= 20
                flagged_events.append("Face not visible - Possible cheating")
                print("ALERT: Face not visible! -20 points")
    else:
        cv2.putText(frame, "Status: Monitoring", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


    cv2.putText(frame, f"Integrity Score: {score}/100", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow("AI Proctor - Live Detection v3", frame)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n--- FINAL REPORT ---")
print(f"Final Score: {score}/100")
if score < 80:
    print("STATUS: FLAGGED FOR REVIEW")
    for event in flagged_events:
        print(f" - {event}")
else:
    print("STATUS: CLEAR")