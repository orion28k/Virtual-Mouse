import cv2
import numpy as np
import mediapipe as mp
import time
import pynput
import screeninfo
#from pocketsphinx import LiveSpeech
import speech_recognition as sr

##########################
wCam, hCam = 640, 480
frameR = 120  # Frame Reduction
smoothening = 7
#########################

# Applications
# Control Powerpoint presentation
# AR Glasses
# Lazarus Girl


pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0


mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=1)
mouse = pynput.mouse.Controller()
keyboard = pynput.keyboard.Controller()

#print(screenWidth, screenHeight)

px, py = 0, 0  # Index finger tip
index_mcp_x, index_mcp_y = 0, 0  # Landmark 5
index_pip_x, index_pip_y = 0, 0  # Landmark 6
thumb_x, thumb_y = 0, 0  # Thumb tip
pix, piy = 0, 0  # Pinky tip
pimx, pimy = 0, 0  # Pinky joint

pressb = False
speak = False
inbounds = False

doVoiceType = False

drag_active = False
drag_threshold = 40
drag_release_threshold = 55
click_threshold = 35
click_touch_active = False

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access a webcam on indices 1 or 0.")

cap.set(3,  wCam)
cap.set(4, hCam)


screen = screeninfo.get_monitors()[0]  # Retrieves the primary screen's dimensions
screenWidth, screenHeight = screen.width, screen.height
# screenWidth, screenHeight = pyautogui.size()


def listen_for_audio():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening... Press 'spacebar' to stop recording.")
        try:
            audio_data = recognizer.listen(source, timeout=None)
            print("Processing audio...")
            text = recognizer.recognize_google(audio_data)
            print("You said:", text)

            if str.lower(text) == "stop program":
                exit()
            else:
                return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")


while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    if not success or img is None or img.size == 0:
        print("Camera frame grab failed, retrying...")
        time.sleep(0.1)
        continue
    inbounds = False
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # 2. Create screen bounds
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)

    # 3. Get the tip of the index and middle fingers
    index_detected = False
    index_mcp_detected = False
    index_pip_detected = False
    thumb_detected = False
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                # print(id, cx, cy)

                if id == 8:
                    px, py = int(lm.x * w), int(lm.y * h)
                    index_detected = True
                    if frameR < px < wCam - frameR and frameR < py < hCam - frameR:
                        # print((px, py), (frameR, wCam - frameR), (frameR, hCam - frameR))
                        cv2.circle(img, (px, py), 5, (0, 255, 0), cv2.FILLED)
                        inbounds = True
                    else:
                        inbounds = False
                elif id == 5:
                    index_mcp_x, index_mcp_y = int(lm.x * w), int(lm.y * h)
                    index_mcp_detected = True
                    if inbounds:
                        cv2.circle(img, (index_mcp_x, index_mcp_y), 5, (0, 255, 255), cv2.FILLED)
                elif id == 6:
                    index_pip_x, index_pip_y = int(lm.x * w), int(lm.y * h)
                    index_pip_detected = True
                    if inbounds:
                        cv2.circle(img, (index_pip_x, index_pip_y), 5, (0, 255, 255))
                elif id == 4:
                    thumb_x, thumb_y = int(lm.x * w), int(lm.y * h)
                    thumb_detected = True
                    if inbounds:
                        cv2.circle(img, (thumb_x, thumb_y), 5, (255, 0, 0), cv2.FILLED)
                elif id == 20:
                    pix, piy = int(lm.x * w), int(lm.y * h)
                    if inbounds:
                        cv2.circle(img, (pix, piy), 5, (0, 0, 255))
                elif id == 18:
                    pimx, pimy = int(lm.x * w), int(lm.y * h)
                    if inbounds:
                        cv2.circle(img, (pimx, pimy), 5, (0, 0, 255))

    # 5. Only Index Finger : Moving Mode
    if inbounds and index_detected:
        # 6. Convert Coordinates
        x3 = np.interp(px, (frameR, wCam - frameR), (0, screenWidth))
        y3 = np.interp(py, (frameR, hCam - frameR), (0, screenHeight))

        # 7. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # 8. Move Mouse
        mouse.position = (int(screenWidth - clocX), int(clocY))

        plocX, plocY = clocX, clocY

    # 9. Thumb gesture handling: drag on thumb-to-index-tip, click on thumb-to-index-base joints
    if thumb_detected and index_detected and inbounds:
        drag_distance = np.hypot(px - thumb_x, py - thumb_y)
        if drag_distance < drag_threshold:
            if not drag_active:
                mouse.press(pynput.mouse.Button.left)
                drag_active = True
        elif drag_distance > drag_release_threshold and drag_active:
            mouse.release(pynput.mouse.Button.left)
            drag_active = False
    else:
        if drag_active:
            mouse.release(pynput.mouse.Button.left)
            drag_active = False

    if not drag_active and thumb_detected and inbounds and (index_mcp_detected or index_pip_detected):
        base_distances = []
        if index_mcp_detected:
            base_distances.append(np.hypot(thumb_x - index_mcp_x, thumb_y - index_mcp_y))
        if index_pip_detected:
            base_distances.append(np.hypot(thumb_x - index_pip_x, thumb_y - index_pip_y))
        if base_distances:
            closest_base = min(base_distances)
            if closest_base < click_threshold:
                if not click_touch_active:
                    mouse.click(pynput.mouse.Button.left, 1)
                    click_touch_active = True
            else:
                click_touch_active = False
    else:
        click_touch_active = False

    if doVoiceType:
        # 10 Pinky is up
        if piy < pimy and inbounds:
            # print(ry, rmy)
            pressb = True
        elif piy > pimy or not inbounds:
            pressb = False

        if pressb:
            if not speak:
                text = listen_for_audio()
                if text:
                    keyboard.type(text)

                    if text[-5:] == "enter":
                        keyboard.tap(pynput.keyboard.Key.enter)
            speak = True
        else:
            speak = False

    # 10. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 11. Display
    window_name = "Virtual Mouse"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # explicit window creation
    cv2.imshow(window_name, img)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window_name, 320, 240)

    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break
