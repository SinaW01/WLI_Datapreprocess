#*************************      Crop the black edges of the data from Hospital A        ***************************

import cv2
import numpy as np

def crop_black_left(input_path, output_path, black_threshold=30, black_ratio=0.85, edge_tolerance=1):
    
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read the image. Please check if the path is correct：{input_path}")

    h, w = img.shape[:2]
    print(f"Original image size: width {w} pixels, height {h} pixels")

    if len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)  
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img  

    black_width = 0
    tolerance_count = 0  
    for col in range(w):
        col_pixels = gray[:, col]
        black_pixel_num = np.sum(col_pixels <= black_threshold)
        black_pixel_ratio = black_pixel_num / h

        if black_pixel_ratio >= black_ratio or tolerance_count < edge_tolerance:
            black_width += 1
            if black_pixel_ratio < black_ratio:
                tolerance_count += 1
        else:
            break  

 
    if black_width >= w:
        print("Warning: The image is entirely a black area; no cropping is needed.")
        cropped_img = img
    else:
        print(f"Detected width of the left black area: {black_width} pixels")
        cropped_img = img[:, black_width:]  


    save_flag = cv2.imwrite(output_path, cropped_img)
    if save_flag:
        crop_h, crop_w = cropped_img.shape[:2]
        print(f"The cropped image has been saved to: {output_path}")
        print(f"Cropped image size: width {crop_w} pixels, height {crop_h} pixels")
    else:
        raise ValueError(f"Failed to save the image. Please check the output path permission: {output_path}")


if __name__ == "__main__":
    input_image_path = r"./inputIMG\2.jpg"
    output_image_path = r"./outputIMG\2_output.jpg"
    crop_black_left(input_image_path, output_image_path, black_threshold=30, black_ratio=0.85, edge_tolerance=1)
    