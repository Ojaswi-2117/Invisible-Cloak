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
# Set camera resolution to maximum supported by the device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width to 1920 (Full HD)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height to 1080 (Full HD)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Check the actual resolution after setting
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution set to: {actual_width}x{actual_height}")
time.sleep(2)
ret, background = cap.read()
if ret:
    background = cv2.flip(background, 1)
else:
    print("Error: Could not read background.")
    exit()

# HSV range for green cloak (default)
lower_cloak1 = np.array([40, 90, 60])
upper_cloak1 = np.array([80, 255, 255])

# For red cloak, we will use two ranges (wrapped HSV)
lower_cloak2 = None
upper_cloak2 = None

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
    global lower_cloak1, upper_cloak1, lower_cloak2, upper_cloak2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Create mask for cloak color
    mask1 = cv2.inRange(hsv, lower_cloak1, upper_cloak1)
    if lower_cloak2 is not None and upper_cloak2 is not None:
        mask2 = cv2.inRange(hsv, lower_cloak2, upper_cloak2)
        mask_red = cv2.bitwise_or(mask1, mask2)
    else:
        mask_red = mask1
    # Skin mask in YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask_skin = cv2.inRange(ycrcb, lower_skin, upper_skin)
    # Remove skin pixels from cloak mask (optional)
    mask = cv2.bitwise_and(mask_red, cv2.bitwise_not(mask_skin))
    # Morphological operations to clean mask
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

# --- Select Cloak Color Function ---
color_selection_mode = False
current_frame_for_callback = None
status_messages = []
exclude_skin_tones = True  # Toggle skin tone exclusion

def select_cloak_color(event, x, y, flags, param):
    global lower_cloak1, upper_cloak1, lower_cloak2, upper_cloak2, color_selection_mode, current_frame_for_callback, status_messages, exclude_skin_tones
    if color_selection_mode and event == cv2.EVENT_LBUTTONDOWN:
        if current_frame_for_callback is not None:
            bgr_color = current_frame_for_callback[y, x]
            hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv_color
            status_messages.append(f"Clicked coordinates: ({x},{y})")
            status_messages.append(f"Selected Color: BGR={bgr_color}, HSV=({h},{s},{v})")
            s_threshold_achromatic = 30
            v_threshold_white = 230
            v_threshold_black = 40
            skin_lower = np.array([0, 40, 50])
            skin_upper = np.array([25, 170, 255])
            new_lower1 = None
            new_upper1 = None
            new_lower2 = None
            new_upper2 = None
            if s < s_threshold_achromatic:
                # Achromatic colors (white, gray, black)
                status_messages.append("-> Achromatic color detected (Low Saturation).")
                if v >= v_threshold_white:
                    status_messages.append("  -> Treating as WHITE")
                    new_lower1 = np.array([0, 0, v_threshold_white - 20])
                    new_upper1 = np.array([179, s_threshold_achromatic, 255])
                elif v <= v_threshold_black:
                    status_messages.append("  -> Treating as BLACK")
                    new_lower1 = np.array([0, 0, 0])
                    new_upper1 = np.array([179, s_threshold_achromatic, v_threshold_black + 20])
                else:
                    status_messages.append("  -> Treating as GRAY")
                    v_delta = 25
                    new_lower1 = np.array([0, 0, max(0, v - v_delta)])
                    new_upper1 = np.array([179, s_threshold_achromatic, min(255, v + v_delta)])
                lower_cloak2 = None
                upper_cloak2 = None
            else:
                # Chromatic colors
                status_messages.append("-> Chromatic color detected (High Saturation).")
                if exclude_skin_tones:
                    if skin_lower[0] <= h <= skin_upper[0] and skin_lower[1] <= s <= skin_upper[1] and skin_lower[2] <= v <= skin_upper[2]:
                        status_messages.append("  -> Detected skin tone. Excluding from cloak color.")
                        color_selection_mode = False
                        return
                hue_range = 6
                if h < hue_range or h > 179 - hue_range:
                    status_messages.append("  -> Treating as RED (Wrapped Hue)")
                    new_lower1 = np.array([0, 70, 60])
                    new_upper1 = np.array([hue_range, 255, 255])
                    new_lower2 = np.array([179 - hue_range, 70, 60])
                    new_upper2 = np.array([179, 255, 255])
                    lower_cloak2 = new_lower2
                    upper_cloak2 = new_upper2
                else:
                    lower_h = max(0, h - hue_range)
                    upper_h = min(179, h + hue_range)
                    new_lower1 = np.array([lower_h, 70, 60])
                    new_upper1 = np.array([upper_h, 255, 255])
                    lower_cloak2 = None
                    upper_cloak2 = None
            # Validate and assign
            if new_lower1 is not None and new_upper1 is not None:
                for i in range(3):
                    if new_lower1[i] > new_upper1[i]:
                        new_upper1[i] = new_lower1[i]
                lower_cloak1 = new_lower1
                upper_cloak1 = new_upper1
                status_messages.append(f"New Cloak HSV Range Set: Lower1={lower_cloak1}, Upper1={upper_cloak1}")
                if lower_cloak2 is not None and upper_cloak2 is not None:
                    status_messages.append(f"New Cloak HSV Range Set: Lower2={lower_cloak2}, Upper2={upper_cloak2}")
                color_selection_mode = False
                status_messages.append("Cloak color set. Press 'k' again to select a new color.")
            else:
                status_messages.append("Error: Could not determine a valid HSV range. Color not changed.")
                color_selection_mode = False
        else:
            status_messages.append("Error: Frame not available for color selection.")
            color_selection_mode = False

# --- MAIN LOOP ---
window_name = "Invisibility Cloak & Meet Mode"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, select_cloak_color)

instructions = [
    "Instructions:",
    "g - Toggle Google Meet Background",
    "1-3 - Select Preloaded Background",
    "u - Upload Custom Background",
    "s - Snap Real-Time Background",
    "r - Reset to Default Settings",
    "a-f - Select Cloak Effect Modes",
    "k - Select Cloak Color",
    "t - Toggle Skin Tone Exclusion",
    "q - Quit"
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    current_frame_for_callback = frame.copy()
    if bg_mode == 'custom':
        gm_bg = get_google_meet_background(google_meet_bg_option, frame.shape)
        with_gm = apply_google_meet_effect(frame, gm_bg)
        final = cloak_effect(with_gm, gm_bg, cloak_effect_mode)
    else:
        final = cloak_effect(frame, background, cloak_effect_mode)

    # Display instructions and status messages on the screen
    y_offset = 30
    for i, line in enumerate(instructions):
        cv2.putText(final, line, (10, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display status messages below instructions
    for j, msg in enumerate(status_messages[-10:]):  # Show only the last 10 messages
        cv2.putText(final, msg, (10, y_offset + len(instructions) * 30 + j * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if color_selection_mode:
        cv2.putText(final, "Click on the desired cloak color", (30, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, final)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        bg_mode = 'real'
        cloak_effect_mode = 'none'
        # Reset cloak color to default green
        lower_cloak1 = np.array([40, 90, 60])
        upper_cloak1 = np.array([80, 255, 255])
        lower_cloak2 = None
        upper_cloak2 = None
        status_messages.append("Cloak color reset to default Green.")
    elif key == ord('g'):
        bg_mode = 'custom'
        google_meet_bg_option = 'realtime'
        status_messages.append("Google Meet Background toggled.")
    elif key == ord('1'):
        google_meet_bg_option = 'image1'
        status_messages.append("Selected Preloaded Background 1.")
    elif key == ord('2'):
        google_meet_bg_option = 'image2'
        status_messages.append("Selected Preloaded Background 2.")
    elif key == ord('3'):
        google_meet_bg_option = 'image3'
        status_messages.append("Selected Preloaded Background 3.")
    elif key == ord('u'):
        path = input("Enter path for uploaded background image: ")
        if os.path.exists(path):
            selected_bg = cv2.imread(path)
            google_meet_bg_option = 'upload'
            status_messages.append("Custom Background Uploaded.")
    elif key == ord('s'):
        temp_ret, temp_frame = cap.read()
        if temp_ret:
            captured_real_bg_cloak = cv2.flip(temp_frame, 1)
            status_messages.append("Captured background for cloak mode 'b'")
    elif key in [ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f')]:
        cloak_effect_mode = chr(key)
        status_messages.append(f"Cloak Effect Mode set to '{chr(key)}'.")
        if key == ord('e'):
            path = input("Enter path for custom image/video for cloak: ")
            if os.path.exists(path):
                if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    custom_cloak = cv2.imread(path)
                elif path.lower().endswith(('.mp4', '.avi')):
                    custom_cap = cv2.VideoCapture(path)
                    if custom_cap.isOpened():
                        custom_cloak = [custom_cap]
                status_messages.append("Custom Cloak Image/Video Loaded.")
    elif key == ord('k'):
        color_selection_mode = True
        status_messages.append("Color selection mode ACTIVE. Click on the cloak color in the window.")
    elif key == ord('t'):
        exclude_skin_tones = not exclude_skin_tones
        status_messages.append(f"Skin tone exclusion {'enabled' if exclude_skin_tones else 'disabled'}.")

# Cleanup
cap.release()
for v in cloak_videos.values():
    v.release()
if isinstance(custom_cloak, list) and custom_cloak[0].isOpened():
    custom_cloak[0].release()
cv2.destroyAllWindows()