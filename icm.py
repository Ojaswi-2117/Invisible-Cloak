import cv2
import numpy as np
import os
import time
import mediapipe as mp

# --- CONFIG ---
BACKGROUND_FOLDER = "backgrounds"
CLOAK_EFFECTS_FOLDER = "cloak_effects"
os.makedirs(BACKGROUND_FOLDER, exist_ok=True)
os.makedirs(CLOAK_EFFECTS_FOLDER, exist_ok=True)

# --- INIT ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
time.sleep(2)

ret, background = cap.read()
if ret:
    background = cv2.flip(background, 1)
else:
    print("Error: Could not read background.")
    exit()

# HSV range for green cloak
lower_green = np.array([40, 90, 60])
upper_green = np.array([80, 255, 255])

# Load animation effects
cloak_videos = {
    'video1': cv2.VideoCapture(os.path.join(CLOAK_EFFECTS_FOLDER, "blueflame.mp4")),
    'video2': cv2.VideoCapture(os.path.join(CLOAK_EFFECTS_FOLDER, "dsportal.mp4"))
}

# MediaPipe Selfie Segmentation
mp_selfie = mp.solutions.selfie_segmentation
selfie = mp_selfie.SelfieSegmentation(model_selection=1)

def get_person_mask(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = selfie.process(rgb)
    if result.segmentation_mask is None:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    condition = result.segmentation_mask > 0.5
    return np.where(condition, 255, 0).astype(np.uint8)

# --- Background Setup ---
selected_bg = None
custom_cloak = None
bg_mode = 'real'
cloak_effect_mode = 'none'
captured_real_bg_cloak = None
google_meet_bg_option = 'realtime'

# Load backgrounds
google_meet_backgrounds = {}
for i, f in enumerate([f for f in os.listdir(BACKGROUND_FOLDER) if f.endswith(('.jpg', '.png'))][:3]):
    path = os.path.join(BACKGROUND_FOLDER, f)
    img = cv2.imread(path)
    if img is not None:
        google_meet_backgrounds[str(i+1)] = img
        print(f"Loaded background {i+1}: {f}")

def resize_bg(img, shape):
    if img is None or img.size == 0:
        return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    return cv2.resize(img, (shape[1], shape[0]))

def get_google_meet_background(option, shape):
    if option == 'realtime':
        ret, real_bg = cap.read()
        return cv2.flip(real_bg, 1) if ret else np.zeros(shape, dtype=np.uint8)
    elif option in ['image1', 'image2', 'image3']:
        key = option[-1]
        if key in google_meet_backgrounds:
            return resize_bg(google_meet_backgrounds[key], shape)
    elif option == 'upload' and selected_bg is not None:
        return resize_bg(selected_bg, shape)
    return get_google_meet_background('realtime', shape)

def apply_cartoon_effect_ghibli(img):
    # Ghibli-style: heavy bilateral filtering and stylized edges
    img_color = img.copy()
    for _ in range(3):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=200, sigmaSpace=200)
    img_edge = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img_color, 255 - img_edge)
    return cartoon

def cloak_effect(frame, bg, mode):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

    shape = frame.shape
    modified_frame = frame.copy()

    if mode == 'a':
        cloak_area = cv2.bitwise_and(bg, bg, mask=mask)
        inv_mask = cv2.bitwise_not(mask)
        normal_area = cv2.bitwise_and(frame, frame, mask=inv_mask)
        modified_frame = cv2.add(cloak_area, normal_area)
    elif mode == 'b' and captured_real_bg_cloak is not None:
        resized = resize_bg(captured_real_bg_cloak, shape)
        cloak_area = cv2.bitwise_and(resized, resized, mask=mask)
        inv_mask = cv2.bitwise_not(mask)
        normal_area = cv2.bitwise_and(frame, frame, mask=inv_mask)
        modified_frame = cv2.add(cloak_area, normal_area)
    elif mode in ['c', 'd']:
        key = 'video1' if mode == 'c' else 'video2'
        success, anim = cloak_videos[key].read()
        if success:
            anim = resize_bg(anim, shape)
            cloak_area = cv2.bitwise_and(anim, anim, mask=mask)
            inv_mask = cv2.bitwise_not(mask)
            normal_area = cv2.bitwise_and(frame, frame, mask=inv_mask)
            modified_frame = cv2.add(cloak_area, normal_area)
        else:
            cloak_videos[key].set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif mode == 'e' and custom_cloak is not None:
        inv_mask = cv2.bitwise_not(mask)
        normal_area = cv2.bitwise_and(frame, frame, mask=inv_mask)
        if isinstance(custom_cloak, list):
            success, anim = custom_cloak[0].read()
            if success:
                anim_resized = resize_bg(anim, shape)
                cloak_area = cv2.bitwise_and(anim_resized, anim_resized, mask=mask)
                modified_frame = cv2.add(cloak_area, normal_area)
        else:
            anim_resized = resize_bg(custom_cloak, shape)
            cloak_area = cv2.bitwise_and(anim_resized, anim_resized, mask=mask)
            modified_frame = cv2.add(cloak_area, normal_area)
    elif mode == 'f':
        ret_bg, real_time_bg = cap.read()
        if ret_bg:
            real_time_bg = cv2.flip(real_time_bg, 1)
            real_time_bg = resize_bg(real_time_bg, shape)

            ghibli = apply_cartoon_effect_ghibli(real_time_bg)
            cartoon_cloak = cv2.bitwise_and(ghibli, ghibli, mask=mask)
            non_cloak = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            modified_frame = cv2.add(non_cloak, cartoon_cloak)
        else:
            modified_frame = frame

    return modified_frame

def apply_google_meet_effect(frame, background):
    person_mask = get_person_mask(frame)
    inv_mask = cv2.bitwise_not(person_mask)
    fg = cv2.bitwise_and(frame, frame, mask=person_mask)
    bg = cv2.bitwise_and(background, background, mask=inv_mask)
    return cv2.add(fg, bg)

# --- MAIN LOOP ---
print("Google Meet + Invisibility Cloak")
print("Keys: g - Google Meet | 1-3/u - BG | a-f - Cloak modes | s - Snap BG | r - Reset | q - Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if bg_mode == 'custom':
        gm_bg = get_google_meet_background(google_meet_bg_option, frame.shape)
        with_gm = apply_google_meet_effect(frame, gm_bg)
        final = cloak_effect(with_gm, gm_bg, cloak_effect_mode)
    else:
        final = cloak_effect(frame, background, cloak_effect_mode)

    cv2.imshow("Invisibility Cloak & Meet Mode", final)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        bg_mode = 'real'
        cloak_effect_mode = 'none'
    elif key == ord('g'):
        bg_mode = 'custom'
        google_meet_bg_option = 'realtime'
    elif key == ord('1'):
        google_meet_bg_option = 'image1'
    elif key == ord('2'):
        google_meet_bg_option = 'image2'
    elif key == ord('3'):
        google_meet_bg_option = 'image3'
    elif key == ord('u'):
        path = input("Enter path for uploaded background image: ")
        if os.path.exists(path):
            selected_bg = cv2.imread(path)
            google_meet_bg_option = 'upload'
    elif key == ord('s'):
        temp_ret, temp_frame = cap.read()
        if temp_ret:
            captured_real_bg_cloak = cv2.flip(temp_frame, 1)
            print("Captured background for cloak mode 'b'")
    elif key in [ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f')]:
        cloak_effect_mode = chr(key)
        if key == ord('e'):
            path = input("Enter path for custom image/video for cloak: ")
            if os.path.exists(path):
                if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    custom_cloak = cv2.imread(path)
                elif path.lower().endswith(('.mp4', '.avi')):
                    custom_cap = cv2.VideoCapture(path)
                    if custom_cap.isOpened():
                        custom_cloak = [custom_cap]

# Cleanup
cap.release()
for v in cloak_videos.values():
    v.release()
if isinstance(custom_cloak, list) and custom_cloak[0].isOpened():
    custom_cloak[0].release()
cv2.destroyAllWindows()
