import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define hand connections manually
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm connections
])

# Create HandLandmarker with Tasks API
base_options = python.BaseOptions(model_asset_path='/home/neal/Desktop/gestureverse/mediapipe/ai_mouse/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)


def is_pinch(landmark_list):
    """Detect pinch gesture."""
    if len(landmark_list) < 21:
        return False
    thumb_tip = landmark_list[4]
    index_tip = landmark_list[8]
    distance = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
    return distance < 0.05


def is_fist(landmark_list):
    """Detect fist gesture."""
    if len(landmark_list) < 21:
        return False
    all_curled = (
        landmark_list[8][1] > landmark_list[6][1] and
        landmark_list[12][1] > landmark_list[10][1] and
        landmark_list[16][1] > landmark_list[14][1] and
        landmark_list[20][1] > landmark_list[18][1]
    )
    return all_curled


def is_index_finger_extended(landmark_list):
    """Detect only index finger extended."""
    if len(landmark_list) < 21:
        return False
    index_extended = landmark_list[8][1] < landmark_list[6][1]
    middle_curled = landmark_list[12][1] > landmark_list[10][1]
    ring_curled = landmark_list[16][1] > landmark_list[14][1]
    pinky_curled = landmark_list[20][1] > landmark_list[18][1]
    return index_extended and middle_curled and ring_curled and pinky_curled


def get_hand_label(handedness):
    """Get hand label."""
    if handedness and len(handedness) > 0:
        return handedness[0].category_name
    return "Unknown"


def draw_info_panel(frame):
    """Draw instruction panel on frame."""
    panel_height = 200
    panel = frame.copy()
    
    # Semi-transparent overlay
    overlay = panel[0:panel_height, :].copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, panel[0:panel_height, :], 0.3, 0, panel[0:panel_height, :])
    
    # Title
    cv2.putText(panel, "GESTURE PRACTICE MODE", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Instructions
    y_offset = 70
    line_height = 30
    
    gestures = [
        ("Pinch (any hand)", "W", (0, 255, 0)),
        ("Left Fist", "A", (255, 0, 0)),
        ("Right Fist", "D", (0, 0, 255)),
        ("Right Index", "S", (0, 255, 255))
    ]
    
    for gesture, key, color in gestures:
        cv2.putText(panel, f"{gesture:20s} -> {key}", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += line_height
    
    return panel


def draw_key_display(frame, detected_keys):
    """Draw large WASD key display showing which keys would be pressed."""
    # Position for key display (bottom right)
    key_size = 80
    spacing = 10
    start_x = frame.shape[1] - (key_size * 3 + spacing * 4)
    start_y = frame.shape[0] - (key_size * 2 + spacing * 3)
    
    # Define key positions (WASD layout)
    keys = {
        'w': (start_x + key_size + spacing, start_y),
        'a': (start_x, start_y + key_size + spacing),
        's': (start_x + key_size + spacing, start_y + key_size + spacing),
        'd': (start_x + (key_size + spacing) * 2, start_y + key_size + spacing)
    }
    
    for key, (x, y) in keys.items():
        # Color: Green if active, gray if not
        color = (0, 255, 0) if key in detected_keys else (100, 100, 100)
        text_color = (255, 255, 255) if key in detected_keys else (150, 150, 150)
        
        # Draw key background
        cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), color, -1)
        cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), (255, 255, 255), 2)
        
        # Draw key letter
        cv2.putText(frame, key.upper(), (x + 20, y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)


def draw_landmarks(frame, hand_landmarks, hand_label):
    """Draw hand landmarks."""
    if hand_landmarks:
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_point = hand_landmarks[connection[0]]
            end_point = hand_landmarks[connection[1]]
            start_pixel = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
            end_pixel = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
            color = (255, 0, 0) if hand_label == "Left" else (0, 0, 255)
            cv2.line(frame, start_pixel, end_pixel, color, 2)
        
        # Draw landmarks
        for idx, landmark in enumerate(hand_landmarks):
            x_pixel = int(landmark.x * frame.shape[1])
            y_pixel = int(landmark.y * frame.shape[0])
            if idx in [4, 8, 12, 16, 20]:
                color = (255, 255, 0)
                radius = 8
            else:
                color = (255, 0, 0) if hand_label == "Left" else (0, 0, 255)
                radius = 5
            cv2.circle(frame, (x_pixel, y_pixel), radius, color, -1)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_timestamp_ms = 0
    
    print("=" * 60)
    print("GESTURE PRACTICE MODE")
    print("=" * 60)
    print("\nPractice your gestures and see which keys they trigger!")
    print("This mode DOES NOT actually press keys - it's for practice only.")
    print("\nPress 'q' to quit")
    print("=" * 60)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Draw instruction panel
            frame = draw_info_panel(frame)
            
            # Convert frame to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Detect hand landmarks
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33
            
            detected_keys = set()
            
            if detection_result.hand_landmarks and detection_result.handedness:
                for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    hand_label = get_hand_label(detection_result.handedness[i] if i < len(detection_result.handedness) else None)
                    
                    # Draw landmarks
                    draw_landmarks(frame, hand_landmarks, hand_label)
                    
                    # Convert to list
                    landmark_list = [(lm.x, lm.y) for lm in hand_landmarks]
                    
                    # Check gestures
                    if is_pinch(landmark_list):
                        detected_keys.add('w')
                        cv2.putText(frame, f"{hand_label}: PINCH -> W", (50, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    elif is_fist(landmark_list):
                        if hand_label == "Left":
                            detected_keys.add('a')
                            cv2.putText(frame, "LEFT FIST -> A", (50, 280), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        elif hand_label == "Right":
                            detected_keys.add('d')
                            cv2.putText(frame, "RIGHT FIST -> D", (50, 280), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    elif is_index_finger_extended(landmark_list):
                        if hand_label == "Right":
                            detected_keys.add('s')
                            cv2.putText(frame, "RIGHT INDEX -> S", (50, 320), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Show hand count
                cv2.putText(frame, f"Hands Detected: {len(detection_result.hand_landmarks)}", 
                           (frame.shape[1] - 300, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw WASD key display
            draw_key_display(frame, detected_keys)
            
            # Show practice mode label
            cv2.putText(frame, "PRACTICE MODE - No keys actually pressed", 
                       (frame.shape[1] - 550, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Gesture Practice Mode', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\nPractice session ended")


if __name__ == '__main__':
    main()