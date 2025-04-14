import cv2
import numpy as np
import os
from datetime import datetime
import subprocess
def iterative_superglue_stitch(start_idx, end_idx, input_dir, output_dir, temp_txt='temp_pairs.txt'):
    """
    Iteratively stitch images using SuperGlue from start_idx to end_idx (inclusive).
    
    Args:
        start_idx (int): Starting image index (e.g., 1 for '1.jpg')
        end_idx (int): Ending image index (inclusive)
        input_dir (str): Directory containing input images
        output_dir (str): Directory to store output images and match files
        temp_txt (str): Temporary SuperGlue pairs file
    """
    def get_img_name(idx): return f"{idx}.jpg"

    def load_image(idx): 
        return cv2.imread(os.path.join(input_dir, get_img_name(idx)))

    def generate_pair_txt(file_path, img1, img2):
        with open(file_path, 'w') as f:
            f.write(f"{img1} {img2}\n")

    def run_superglue(txt_filename):
        match_script = os.path.abspath("match_pairs.py")
        command = [
            "python", match_script,
            "--resize", "-1",
            "--superglue", "outdoor",
            "--max_keypoints", "2048",
            "--nms_radius", "5",
            "--resize_float",
            "--input_dir", input_dir,
            "--input_pairs", txt_filename,
            "--output_dir", output_dir,
            "--viz",
            "--keypoint_threshold", "0.05",
            "--match_threshold", "0.9"
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print("SuperGlue failed!", result.stderr)
            return False
        return True

    def load_npz_matches(file_name, output_dir):
        npz = np.load(os.path.join(output_dir, file_name))
        matches = npz['matches']
        valid = matches > -1
        kp1 = npz['keypoints0'][valid]
        kp2 = npz['keypoints1'][matches[valid]]
        return kp1, kp2

    def crop_black(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]

    def stitch_from_numpy_matches(img1, img2, kp1, kp2):
        pts1 = np.float32(kp1).reshape(-1, 1, 2)
        pts2 = np.float32(kp2).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
        if H is None:
            raise ValueError("Homography failed.")
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners = np.concatenate((np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2), warped_corners), axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        translation = [-xmin, -ymin]
        H_translation = np.array([[1, 0, translation[0]],
                                  [0, 1, translation[1]],
                                  [0, 0, 1]])
        result = cv2.warpPerspective(img2, H_translation @ H, (xmax - xmin, ymax - ymin))
        result[translation[1]:h1+translation[1], translation[0]:w1+translation[0]] = img1
        return crop_black(result)

    stitched_img = load_image(start_idx)
    stitched_name = f"{start_idx}"

    for i in range(start_idx + 1, end_idx + 1):
        next_img_name = get_img_name(i)
        curr_img_path = os.path.join(output_dir, f"{stitched_name}_{i}.jpg")

        generate_pair_txt(temp_txt, f"{stitched_name}.jpg", next_img_name)

        temp_stitch_path = os.path.join(input_dir, f"{stitched_name}.jpg")
        cv2.imwrite(temp_stitch_path, stitched_img)

        npz_filename = f"{stitched_name}_{next_img_name[:-4]}_matches.npz"
        print(f"Running SuperGlue between {stitched_name}.jpg and {next_img_name}")
        success=1
        if not os.path.exists(f"{output_dir}/{npz_filename}"):
          success = run_superglue(temp_txt)
        if not success:
            print(f"Skipping {i} due to match failure.")
            continue

        kp1, kp2 = load_npz_matches(npz_filename, output_dir)
        next_img = load_image(i)

        stitched_img = stitch_from_numpy_matches(stitched_img, next_img, kp1, kp2)
        stitched_name += f"_{i}"
        cv2.imwrite(curr_img_path, stitched_img)
        print(f"Saved stitched image: {curr_img_path}")

    return stitched_img