import cv2
import numpy as np


def crop_black_border_from_center(
    input_path,
    output_path,
    black_threshold=28,
    center_ratio=0.4,
    min_bright_count=3,
    extra_margin=2
):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {input_path}")

    if len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape

    # ────────────────────────────────
    # Calculate the range of the middle section (avoiding the four corners)
    # ────────────────────────────────
    center_h_start = int(h * (0.5 - center_ratio / 2))
    center_h_end   = int(h * (0.5 + center_ratio / 2))
    center_w_start = int(w * (0.5 - center_ratio / 2))
    center_w_end   = int(w * (0.5 + center_ratio / 2))

    center_h_start = max(0, center_h_start)
    center_h_end   = min(h, center_h_end)
    center_w_start = max(0, center_w_start)
    center_w_end   = min(w, center_w_end)

    # ────────────────────────────────
    #  Left side: From left to right, only look at the middle height segment of each column.
    # ────────────────────────────────
    x1 = 0
    for x in range(w):
        col_mid = gray[center_h_start:center_h_end, x]
        bright_count = np.sum(col_mid > black_threshold)
        if bright_count >= min_bright_count:
            x1 = x
            break

    # ────────────────────────────────
    # Right side: From right to left, only look at the middle height segment of each column.
    # ────────────────────────────────
    x2 = w - 1
    for x in range(w - 1, -1, -1):
        col_mid = gray[center_h_start:center_h_end, x]
        bright_count = np.sum(col_mid > black_threshold)
        if bright_count >= min_bright_count:
            x2 = x
            break

    # ────────────────────────────────
    # Upper side: From top to bottom, only look at the middle width segment of each row.
    # ────────────────────────────────
    y1 = 0
    for y in range(h):
        row_mid = gray[y, center_w_start:center_w_end]
        bright_count = np.sum(row_mid > black_threshold)
        if bright_count >= min_bright_count:
            y1 = y
            break

    # ────────────────────────────────
    #  Lower side: From bottom to top, only look at the middle width segment of each row.
    # ────────────────────────────────
    y2 = h - 1
    for y in range(h - 1, -1, -1):
        row_mid = gray[y, center_w_start:center_w_end]
        bright_count = np.sum(row_mid > black_threshold)
        if bright_count >= min_bright_count:
            y2 = y
            break


    if y1 >= y2 or x1 >= x2 or (y2 - y1 < 30) or (x2 - x1 < 30):
        print("The valid area is too small; the original image is retained.")
        cv2.imwrite(output_path, img)
        return


    y1 = max(0, y1 + extra_margin)
    x1 = max(0, x1 + extra_margin)
    y2 = min(h - 1, y2 - extra_margin)
    x2 = min(w - 1, x2 - extra_margin)

    cropped = img[y1:y2+1, x1:x2+1]

    success = cv2.imwrite(output_path, cropped)
    if success:
        print(f"Saved successfully: {output_path} Size {cropped.shape[1]}×{cropped.shape[0]}")
    else:
        print(f"Failed to save: {output_path}")



if __name__ == "__main__":
    crop_black_border_from_center(
        "./outputIMG_37/37/37_frame_00546.jpg",
        "./outputIMG_37/37_NBI/37_frame_00546.png",
        black_threshold=28,
        center_ratio=0.45,
        min_bright_count=4,
        extra_margin=1
    )