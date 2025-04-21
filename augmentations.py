# -*- coding: utf-8 -*-
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# --- Keystone Augmentation ---
def random_keystone_adjustment(image_pil, num_degrees=6):
    """Performs a keystone perspective shift on a PIL Image."""
    try:
        orig_img = np.array(image_pil.convert('RGB'))
        orig_h, orig_w = orig_img.shape[:2]
        pad_x, pad_y = int(0.2 * orig_w), int(0.2 * orig_h)

        padded_img = cv2.copyMakeBorder(orig_img, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REFLECT_101)
        padded_h, padded_w = padded_img.shape[:2]

        max_shift_ratio = np.tan(np.radians(num_degrees))
        max_shift_x, max_shift_y = int(padded_w * max_shift_ratio), int(padded_h * max_shift_ratio)

        shift_x = [random.randint(-max_shift_x, max_shift_x) for _ in range(4)]
        shift_y = [random.randint(-max_shift_y, max_shift_y) for _ in range(4)]

        src_pts = np.float32([[0, 0], [padded_w - 1, 0], [padded_w - 1, padded_h - 1], [0, padded_h - 1]])
        dst_pts = np.float32([
            [0 + shift_x[0], 0 + shift_y[0]], [padded_w - 1 + shift_x[1], 0 + shift_y[1]],
            [padded_w - 1 + shift_x[2], padded_h - 1 + shift_y[2]], [0 + shift_x[3], padded_h - 1 + shift_y[3]]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(padded_img, M, (padded_w, padded_h), borderMode=cv2.BORDER_REFLECT_101)

        center_x, center_y = padded_w // 2, padded_h // 2
        x1, y1 = center_x - (orig_w // 2), center_y - (orig_h // 2)
        x2, y2 = x1 + orig_w, y1 + orig_h
        final_crop = warped[y1:y2, x1:x2]

        return Image.fromarray(final_crop)
    except Exception as e:
        print(f"Warning: Error during keystone adjustment: {e}. Returning original image.")
        return image_pil

# --- Main Augmentation Function ---
def apply_augmentations(image_bgr, cnn_type=1):
    """
    Applies random augmentations to an image (expects BGR format from cv2.imread).
    Returns a PIL Image (RGB) and a flip status (for CNN2).
    Optimized slightly for less conversion.
    """
    image = image_bgr.copy()
    flipped = False

    # 1. Horizontal flip (50% chance)
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        flipped = True

    # 2. Court Color Augmentation (66% chance)
    if random.random() < 0.66:
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_court = np.array([0, 30, 30])
            upper_court = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower_court, upper_court)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            target_hue_base = random.choice([70, 110]) # Greenish or Bluish
            hue_shift = random.randint(-5, 5)
            target_hue = (target_hue_base + hue_shift) % 180
            hsv[:, :, 0] = np.where(mask == 255, target_hue, hsv[:, :, 0])
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e:
            print(f"Warning: Color augmentation failed: {e}")


    # Convert to PIL RGB ONCE before potential PIL-based augmentations
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Warning: Failed converting to PIL: {e}. Using original BGR converted.")
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)) # Fallback


    # 3. Brightness/Contrast Jitter (40% chance) - operates on PIL
    if random.random() < 0.4:
        try:
            enhancer = ImageEnhance.Brightness(image_pil)
            image_pil = enhancer.enhance(random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Contrast(image_pil)
            image_pil = enhancer.enhance(random.uniform(0.8, 1.2))
        except Exception as e:
            print(f"Warning: Brightness/Contrast jitter failed: {e}")


    # 4. Keystone Augmentation (50% chance) - operates on PIL
    if random.random() < 0.5:
        image_pil = random_keystone_adjustment(image_pil)

    # Return the final augmented PIL image (RGB) and flip status
    return image_pil, flipped