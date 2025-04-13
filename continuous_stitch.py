import cv2
import numpy as np
import os
import subprocess

def run_superglue_on_all_pairs(start_idx, end_idx, input_dir, output_dir, temp_txt='temp_pairs.txt'):
    """
    Precompute SuperGlue matches for all adjacent pairs (i, i+1).
    """
    print("\n=== Running SuperGlue on all adjacent pairs ===")
    with open(temp_txt, 'w') as f:
        for i in range(start_idx, end_idx):
            img1 = f"{i}.jpg"
            img2 = f"{i+1}.jpg"
            f.write(f"{img1} {img2}\n")

    command = [
        "python", "match_pairs.py",
        "--resize", "-1",
        "--superglue", "outdoor",
        "--max_keypoints", "2048",
        "--nms_radius", "5",
        "--resize_float",
        "--input_dir", input_dir,
        "--input_pairs", temp_txt,
        "--output_dir", output_dir,
        "--viz",
        "--keypoint_threshold", "0.05",
        "--match_threshold", "0.9"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("SuperGlue failed:\n" + result.stderr)
    print("✅ SuperGlue matches generated for all adjacent pairs.")


def stitch_images_sequentially(start_idx, end_idx, input_dir, output_dir):
    def load_image(name): return cv2.imread(os.path.join(input_dir, name))

    def crop_black(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]

    def load_npz_matches(img1, img2):
        filename = f"{img1[:-4]}_{img2[:-4]}_matches.npz"
        npz = np.load(os.path.join(output_dir, filename))
        matches = npz['matches']
        valid = matches > -1
        kp1 = npz['keypoints0'][valid]
        kp2 = npz['keypoints1'][matches[valid]]
        return kp1, kp2

    def stitch(img1, img2, kp1, kp2):
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

    stitched = load_image(f"{start_idx}.jpg")
    for i in range(start_idx, end_idx):
        img1_name = f"{i}.jpg"
        img2_name = f"{i+1}.jpg"
        kp1, kp2 = load_npz_matches(img1_name, img2_name)
        img2 = load_image(img2_name)
        print(f"🔗 Stitching {img1_name} + {img2_name}")
        stitched = stitch(stitched, img2, kp1, kp2)
        cv2.imwrite(os.path.join(input_dir, f"stitched_up_to_{i+1}.jpg"), stitched)

    print(f"\n✅ Final stitched image saved as: stitched_up_to_{end_idx}.jpg")
    return stitched


def run_pipeline(start_idx, end_idx, input_dir, output_dir):
    run_superglue_on_all_pairs(start_idx, end_idx, input_dir, output_dir)
    final_img = stitch_images_sequentially(start_idx, end_idx, input_dir, output_dir)
    return final_img

