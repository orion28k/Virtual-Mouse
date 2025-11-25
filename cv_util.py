import mediapipe as mp
import cv2

try:
    import tkinter as tk
except Exception:  # pragma: no cover - tkinter not always available
    tk = None

def get_screen_size(default=(1920, 1080)):
    """
    Determine the primary screen resolution.

    Args:
        default: Tuple[int, int] fallback resolution if tkinter is unavailable.

    Returns:
        Tuple[int, int]: Width and height of the primary display.
    """
    if tk is None:
        return default

    root = tk.Tk()
    root.withdraw()
    try:
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
    finally:
        root.destroy()
    return width, height

# ---------- MediaPipe Hands ---------

def init_hands():
    """
    Initialize and return a MediaPipe Hands instance.
    main.py calls this once and reuses the object.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return hands

def process_hands(img, hands, draw = False,hands_array = [None, None]):
    '''
    Args
    - Image
    - Medipipe Hands object
    - Draw hands?

    Return
    - Hand array [index 0 left hand or index 1 right hand][landmark] or None
    - If draw, draw 
    '''
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hands_array = [None, None]

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if draw:
                draw_hands(img, results.multi_hand_landmarks)

            label = handedness.classification[0].label.lower()  # "left" or "right"

            if label == "left":
                hands_array[0] = hand_landmarks
            elif label == "right":
                hands_array[1] = hand_landmarks

    return hands_array

def draw_hands(img, multi_hand_landmarks):
    mp_drawing = mp.solutions.drawing_utils

    for hand_landmarks in multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, # image
            hand_landmarks, # hand landmarks
            mp.solutions.hands.HAND_CONNECTIONS, # list of index pairs that define the connections
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3), # customize landmarks
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2) # customize lines
        )
