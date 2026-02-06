import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import subprocess

screen_width, screen_height = pyautogui.size()

# Disable PyAutoGUI failsafe
pyautogui.FAILSAFE = False

# Define hand connections manually
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm connections
])

# Create HandLandmarker with Tasks API - NOW DETECTING 2 HANDS
base_options = python.BaseOptions(model_asset_path='/home/neal/Desktop/gestureverse/mediapipe/ai_mouse/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,  # Changed to 2 to detect both hands
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Key press tracking
current_keys = set()
last_action_time = {}
action_cooldown = 0.4  # seconds
ACTION_COOLDOWN = action_cooldown

### me modified on tapping
def tap_key(key):
    """Human-like key tap using xdotool"""
    now = time.time()
    if key in last_action_time and now - last_action_time[key] < ACTION_COOLDOWN:
        return

    subprocess.run(["xdotool", "keydown", key])
    time.sleep(0.05)   # 50ms = human tap
    subprocess.run(["xdotool", "keyup", key])

    last_action_time[key] = now
    print(f"TAP -> {key}")

def press_key(key):
    """Press and hold a key if not already pressed."""
    global current_keys, last_action_time
    
    current_time = time.time()
    
    # Check cooldown for this specific key
    if key in last_action_time:
        if current_time - last_action_time[key] < action_cooldown:
            return
    
    if key not in current_keys:
        # subprocess.run(["xdotool", "keydown", key])
        # pyautogui.keyDown(key)
        current_keys.add(key)
        last_action_time[key] = current_time
        print(f"Pressed: {key}")


def release_key(key):
    """Release a key if currently pressed."""
    global current_keys
    
    if key in current_keys:
        # subprocess.run(["xdotool", "keyup", key])
        # pyautogui.keyUp(key)
        current_keys.remove(key)
        print(f"Released: {key}")


def release_all_keys():
    """Release all currently pressed keys."""
    global current_keys
    
    for key in list(current_keys):
        pyautogui.keyUp(key)
    current_keys.clear()


def is_pinch(landmark_list):
    """Detect pinch gesture - thumb and index close together."""
    if len(landmark_list) < 21:
        return False
    
    thumb_tip = landmark_list[4]
    index_tip = landmark_list[8]
    
    # Calculate distance between thumb and index
    distance = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
    
    return distance < 0.05


def is_fist(landmark_list):
    """Detect fist gesture - all fingers curled."""
    if len(landmark_list) < 21:
        return False
    
    # All fingertips should be lower (higher Y value) than their base joints
    all_curled = (
        landmark_list[8][1] > landmark_list[6][1] and   # Index curled
        landmark_list[12][1] > landmark_list[10][1] and  # Middle curled
        landmark_list[16][1] > landmark_list[14][1] and  # Ring curled
        landmark_list[20][1] > landmark_list[18][1]      # Pinky curled
    )
    
    return all_curled


def is_index_finger_extended(landmark_list):
    """Detect only index finger extended, others curled."""
    if len(landmark_list) < 21:
        return False
    
    # Index extended (tip higher than base)
    index_extended = landmark_list[8][1] < landmark_list[6][1]
    
    # Other fingers curled
    middle_curled = landmark_list[12][1] > landmark_list[10][1]
    ring_curled = landmark_list[16][1] > landmark_list[14][1]
    pinky_curled = landmark_list[20][1] > landmark_list[18][1]
    
    return index_extended and middle_curled and ring_curled and pinky_curled


def get_hand_label(handedness):
    """Get hand label (Left or Right)."""
    if handedness and len(handedness) > 0:
        return handedness[0].category_name
    return "Unknown"


def detect_gesture(frame, hands_data):
    """
    Detect and execute keyboard gestures.
    
    Gesture mapping:
    - Pinch (any hand): W
    - Left hand fist: A
    - Right hand fist: D
    - Right hand index finger: S
    """
    
    if not hands_data:
        # No hands detected - release all keys
        release_all_keys()
        return
    
    # Track which keys should be pressed this frame
    keys_to_press = set()
    
    for hand_landmarks, hand_label in hands_data:
        landmark_list = [(lm.x, lm.y) for lm in hand_landmarks]
        
        # Check gestures
        if is_pinch(landmark_list):
            keys_to_press.add('Up')
            cv2.putText(frame, f"{hand_label}: PINCH -> W", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        elif is_fist(landmark_list):
            if hand_label == "Left":
                keys_to_press.add('Left')
                cv2.putText(frame, "LEFT FIST -> A", (50, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            elif hand_label == "Right":
                keys_to_press.add('Right')
                cv2.putText(frame, "RIGHT FIST -> D", (50, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        elif is_index_finger_extended(landmark_list):
            if hand_label == "Right":
                keys_to_press.add('Down')
                cv2.putText(frame, "RIGHT INDEX -> S", (50, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Press keys that should be active
    for key in keys_to_press:
        tap_key(key)
    
    # Release keys that should not be active
    for key in list(current_keys):
        if key not in keys_to_press:
            release_key(key)


def draw_landmarks_on_frame(frame, hand_landmarks, hand_label):
    """Draw hand landmarks on the frame with hand label."""
    if hand_landmarks:
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            start_pixel = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
            end_pixel = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
            
            # Color based on hand
            color = (255, 0, 0) if hand_label == "Left" else (0, 0, 255)
            cv2.line(frame, start_pixel, end_pixel, color, 2)
        
        # Draw landmarks
        for idx, landmark in enumerate(hand_landmarks):
            x_pixel = int(landmark.x * frame.shape[1])
            y_pixel = int(landmark.y * frame.shape[0])
            
            # Different colors for different parts
            if idx in [4, 8, 12, 16, 20]:  # Fingertips
                color = (255, 255, 0)  # Yellow
                radius = 8
            elif idx == 0:  # Wrist
                color = (0, 255, 255)  # Cyan
                radius = 10
            else:  # Other joints
                color = (255, 0, 0) if hand_label == "Left" else (0, 0, 255)
                radius = 5
            
            cv2.circle(frame, (x_pixel, y_pixel), radius, color, -1)
        
        # Draw hand label
        wrist = hand_landmarks[0]
        label_x = int(wrist.x * frame.shape[1])
        label_y = int(wrist.y * frame.shape[0]) - 20
        cv2.putText(frame, hand_label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_timestamp_ms = 0
    
    print("=" * 60)
    print("KEYBOARD CONTROL WITH HAND GESTURES")
    print("=" * 60)
    print("\nGesture Mapping:")
    print("  🤏 Pinch (any hand)        → W")
    print("  🤛 Left hand fist          → A")
    print("  🤜 Right hand fist         → D")
    print("  ☝️ Right hand index finger → S")
    print("\nPress 'q' to quit")
    print("=" * 60)
    print()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # frame = cv2.flip(frame, 1)
            
            # Convert frame to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Detect hand landmarks
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33  # Assuming ~30 fps
            
            hands_data = []
            
            if detection_result.hand_landmarks and detection_result.handedness:
                # Process each detected hand
                for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    hand_label = get_hand_label(detection_result.handedness[i] if i < len(detection_result.handedness) else None)
                    
                    # Draw landmarks
                    draw_landmarks_on_frame(frame, hand_landmarks, hand_label)
                    
                    # Store hand data for gesture detection
                    hands_data.append((hand_landmarks, hand_label))
                
                # Show hand count
                cv2.putText(frame, f"Hands: {len(detection_result.hand_landmarks)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No hands detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Detect gestures
            detect_gesture(frame, hands_data)
            
            # Show currently pressed keys
            if current_keys:
                keys_text = "Keys: " + ", ".join(sorted(current_keys)).upper()
                cv2.putText(frame, keys_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Keyboard Control - Hand Gestures', frame)
            cv2.setWindowProperty(
                'Keyboard Control - Hand Gestures',
                cv2.WND_PROP_TOPMOST,
                0
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release all keys before exiting
        release_all_keys()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\nAll keys released and cleaned up successfully")


if __name__ == '__main__':
    main()