#**********************                 Crop the black edges of the data from Hospital B (rectangle)                 **************************        

import cv2
import numpy as np


def crop_to_first_nonblack(
    input_path,
    output_path,
    black_threshold=30,
    margin=1
):
    
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read the image: {input_path}")

    if len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape

    # ────────────────────────────────
    #  1. From left to right, find the first column with non-black pixels.
    # ────────────────────────────────
    x1 = 0
    for x in range(w):
        if np.any(gray[:, x] > black_threshold):
            x1 = x
            break

    # ────────────────────────────────
    #  2. From right to left, find the first column with non-black pixels.
    # ────────────────────────────────
    x2 = w - 1
    for x in range(w - 1, -1, -1):
        if np.any(gray[:, x] > black_threshold):
            x2 = x
            break

    # ────────────────────────────────
    #  3.From top to bottom, find the first row with non-black pixels.
    # ────────────────────────────────
    y1 = 0
    for y in range(h):
        if np.any(gray[y, :] > black_threshold):
            y1 = y
            break

    # ────────────────────────────────
    #  4. From bottom to top, find the first row with non-black pixels.
    # ────────────────────────────────
    y2 = h - 1
    for y in range(h - 1, -1, -1):
        if np.any(gray[y, :] > black_threshold):
            y2 = y
            break

    
    if y1 >= y2 or x1 >= x2 or (y2 - y1 < 10) or (x2 - x1 < 10):
        print("Warning: The detected valid area is too small or completely black; the original image is retained.")
        cv2.imwrite(output_path, img)
        return


    y1 = max(0, y1 + margin)
    x1 = max(0, x1 + margin)
    y2 = min(h - 1, y2 - margin)
    x2 = min(w - 1, x2 - margin)

    
    cropped = img[y1:y2+1, x1:x2+1]


    success = cv2.imwrite(output_path, cropped)
    if success:
        print(f"Cropping result saved successfully: {output_path}")
        print(f"Cropped size: {cropped.shape[1]} × {cropped.shape[0]}")
    else:
        print(f"Failed to save: {output_path}")



if __name__ == "__main__":
    input_file  = r"./outputIMG_23/23/23_frame_00608.jpg"
    output_file = r"./outputIMG_23/23_NBI/23_frame_00608_cropped.png"
    
    crop_to_first_nonblack(
        input_file,
        output_file,
        black_threshold=30,  
        margin=1              
    )



    #Cropped size: 1278 × 1022