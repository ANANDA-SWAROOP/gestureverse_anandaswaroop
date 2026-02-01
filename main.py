import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import random
import util
from pynput.mouse import Button, Controller
import time

mouse = Controller()

screen_width, screen_height = pyautogui.size()

# Disable PyAutoGUI failsafe for better control
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

# Create HandLandmarker with Tasks API
base_options = python.BaseOptions(model_asset_path='/home/neal/Desktop/gestureverse/mediapipe/ai_mouse/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.65
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Gesture debouncing
last_click_time = 0
click_cooldown = 0.5  # seconds between clicks


def find_finger_tip(hand_landmarks):
    """Extract index finger tip from hand landmarks."""
    if hand_landmarks:
        # Index finger tip is landmark 8
        index_finger_tip = hand_landmarks[0][8]
        return index_finger_tip
    return None


def move_mouse(index_finger_tip):
    """Move mouse cursor based on index finger tip position."""
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)  # Removed /2 for full screen coverage
        pyautogui.moveTo(x, y, duration=0)  # duration=0 for instant movement


def is_left_click(landmark_list, thumb_index_dist):
    """Detect left click gesture - index finger extended, middle curled, fingers apart."""
    if len(landmark_list) < 21:
        return False
    
    # Index finger extended (tip higher than base)
    index_extended = landmark_list[8][1] < landmark_list[6][1]
    
    # Middle finger curled (tip lower than base)
    middle_curled = landmark_list[12][1] > landmark_list[10][1]
    
    # Thumb and index far apart
    fingers_apart = thumb_index_dist > 0.05
    
    return index_extended and middle_curled and fingers_apart


def is_right_click(landmark_list, thumb_index_dist):
    """Detect right click gesture - middle finger extended, index curled, fingers apart."""
    if len(landmark_list) < 21:
        return False
    
    # Middle finger extended (tip higher than base)
    middle_extended = landmark_list[12][1] < landmark_list[10][1]
    
    # Index finger curled (tip lower than base)
    index_curled = landmark_list[8][1] > landmark_list[6][1]
    
    # Thumb and index far apart
    fingers_apart = thumb_index_dist > 0.05
    
    return middle_extended and index_curled and fingers_apart


def is_double_click(landmark_list, thumb_index_dist):
    """Detect double click gesture - both index and middle extended, fingers apart."""
    if len(landmark_list) < 21:
        return False
    
    # Both index and middle extended
    index_extended = landmark_list[8][1] < landmark_list[6][1]
    middle_extended = landmark_list[12][1] < landmark_list[10][1]
    
    # Ring and pinky curled
    ring_curled = landmark_list[16][1] > landmark_list[14][1]
    pinky_curled = landmark_list[20][1] > landmark_list[18][1]
    
    # Fingers apart
    fingers_apart = thumb_index_dist > 0.05
    
    return index_extended and middle_extended and ring_curled and pinky_curled and fingers_apart


def is_screenshot(landmark_list, thumb_index_dist):
    """Detect screenshot gesture - pinch (thumb and index close together)."""
    if len(landmark_list) < 21:
        return False
    
    # Thumb and index very close together
    pinch = thumb_index_dist < 0.05
    
    return pinch


def is_mouse_move_mode(landmark_list, thumb_index_dist):
    """Detect mouse move mode - closed fist or thumb-index close with other fingers curled."""
    if len(landmark_list) < 21:
        return False
    
    # All fingers curled (fist)
    all_curled = (
        landmark_list[8][1] > landmark_list[6][1] and   # Index curled
        landmark_list[12][1] > landmark_list[10][1] and  # Middle curled
        landmark_list[16][1] > landmark_list[14][1] and  # Ring curled
        landmark_list[20][1] > landmark_list[18][1]      # Pinky curled
    )
    
    return all_curled or thumb_index_dist < 0.05


def detect_gesture(frame, landmark_list, hand_landmarks):
    """Detect and execute gestures based on hand landmarks."""
    global last_click_time
    
    if len(landmark_list) < 21:
        return
    
    # Calculate thumb-index distance
    thumb_tip = landmark_list[4]
    index_tip = landmark_list[8]
    thumb_index_dist = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
    
    current_time = time.time()
    
    # Debug info
    cv2.putText(frame, f"Distance: {thumb_index_dist:.3f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Check gestures in priority order
    if is_screenshot(landmark_list, thumb_index_dist):
        if current_time - last_click_time > click_cooldown:
            try:
                im1 = pyautogui.screenshot()
                label = random.randint(1, 1000)
                im1.save(f'my_screenshot_{label}.png')
                cv2.putText(frame, "SCREENSHOT TAKEN", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                print("Screenshot taken!")
                last_click_time = current_time
            except Exception as e:
                print(f"Screenshot error: {e}")
    
    elif is_double_click(landmark_list, thumb_index_dist):
        if current_time - last_click_time > click_cooldown:
            try:
                pyautogui.doubleClick()
                cv2.putText(frame, "DOUBLE CLICK", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                print("Double click executed!")
                last_click_time = current_time
            except Exception as e:
                print(f"Double click error: {e}")
    
    elif is_left_click(landmark_list, thumb_index_dist):
        if current_time - last_click_time > click_cooldown:
            try:
                pyautogui.click()  # Using pyautogui instead of pynput
                cv2.putText(frame, "LEFT CLICK", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                print("Left click executed!")
                last_click_time = current_time
            except Exception as e:
                print(f"Left click error: {e}")
    
    elif is_right_click(landmark_list, thumb_index_dist):
        if current_time - last_click_time > click_cooldown:
            try:
                pyautogui.rightClick()  # Using pyautogui instead of pynput
                cv2.putText(frame, "RIGHT CLICK", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("Right click executed!")
                last_click_time = current_time
            except Exception as e:
                print(f"Right click error: {e}")
    
    elif is_mouse_move_mode(landmark_list, thumb_index_dist):
        index_finger_tip = find_finger_tip(hand_landmarks)
        move_mouse(index_finger_tip)
        cv2.putText(frame, "MOVING MOUSE", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_landmarks_on_frame(frame, hand_landmarks):
    """Draw all 21 hand landmarks on the frame."""
    if hand_landmarks:
        for landmarks in hand_landmarks:
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                start_pixel = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
                end_pixel = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
                
                cv2.line(frame, start_pixel, end_pixel, (0, 255, 0), 2)
            
            # Draw all 21 landmarks
            for idx, landmark in enumerate(landmarks):
                x_pixel = int(landmark.x * frame.shape[1])
                y_pixel = int(landmark.y * frame.shape[0])
                
                # Different colors for different parts
                if idx in [4, 8, 12, 16, 20]:  # Fingertips
                    color = (255, 0, 0)  # Blue
                    radius = 8
                elif idx == 0:  # Wrist
                    color = (0, 255, 255)  # Yellow
                    radius = 10
                else:  # Other joints
                    color = (0, 0, 255)  # Red
                    radius = 5
                
                cv2.circle(frame, (x_pixel, y_pixel), radius, color, -1)
                
                # Draw landmark index for debugging
                cv2.putText(frame, str(idx), (x_pixel + 10, y_pixel), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_timestamp_ms = 0
    
    print("=" * 50)
    print("Hand Gesture Control Started")
    print("=" * 50)
    print("\nGestures:")
    print("  - Fist/Pinch: Move mouse")
    print("  - Index up + Middle down: Left click")
    print("  - Middle up + Index down: Right click")
    print("  - Index + Middle up (peace sign): Double click")
    print("  - Thumb + Index pinch (close): Screenshot")
    print("\nPress 'q' to quit")
    print("=" * 50)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Convert frame to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Detect hand landmarks
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33  # Assuming ~30 fps
            
            landmark_list = []
            hand_landmarks_list = None
            
            if detection_result.hand_landmarks:
                hand_landmarks_list = detection_result.hand_landmarks
                hand_landmarks = detection_result.hand_landmarks[0]  # Get first hand
                
                # Convert landmarks to list format for compatibility
                for landmark in hand_landmarks:
                    landmark_list.append((landmark.x, landmark.y))
                
                # Show landmark count
                cv2.putText(frame, f"Landmarks: {len(landmark_list)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw all landmarks on frame
                draw_landmarks_on_frame(frame, hand_landmarks_list)
                
                # Detect gestures
                detect_gesture(frame, landmark_list, hand_landmarks_list)
            else:
                cv2.putText(frame, "No hand detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Hand Gesture Control', frame)
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
        print("\nCleaned up successfully")


if __name__ == '__main__':
    main()