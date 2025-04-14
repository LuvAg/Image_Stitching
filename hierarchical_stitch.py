import cv2
import numpy as np
import os
import subprocess

def iterative_superglue_stitch(img1_name, img2_name, input_dir, output_dir, temp_txt='temp_pairs.txt'):
    def load_image(name): return cv2.imread(os.path.join(input_dir, name))

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

    def load_npz_matches(file_name):
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

    # Run SuperGlue matching
    print(f"Generating pair text for {img1_name} and {img2_name}")
    generate_pair_txt(temp_txt, img1_name, img2_name)
    if not os.path.exists(os.path.join(output_dir, f"{img1_name[:-4]}_{img2_name[:-4]}_matches.npz")):
        print(f"Running SuperGlue for {img1_name} and {img2_name}")
        success = run_superglue(temp_txt)
        if not success:
            raise RuntimeError(f"SuperGlue matching failed between {img1_name} and {img2_name}")

    # Load matches and images
    print(f"Loading matches for {img1_name} and {img2_name}")
    kp1, kp2 = load_npz_matches(f"{img1_name[:-4]}_{img2_name[:-4]}_matches.npz")
    img1 = load_image(img1_name)
    img2 = load_image(img2_name)

    # Stitch the images
    print(f"Stitching {img1_name} and {img2_name}")
    stitched_img = stitch_from_numpy_matches(img1, img2, kp1, kp2)
    stitched_name = f"{img1_name[:-4]}_{img2_name[:-4]}.jpg"
    cv2.imwrite(os.path.join(output_dir, stitched_name), stitched_img)  # <--- fixed path
    print(f"Stitched image saved as: {stitched_name}")
    return stitched_name



def hierarchical_stitching(start_idx, end_idx, input_dir, output_dir):
    """
    Perform hierarchical pairwise stitching on images from start_idx to end_idx.
    """
    # Initial image names
    img_list = [f"{i}.jpg" for i in range(start_idx, end_idx + 1)]

    round_num = 0
    while len(img_list) > 1:
        print(f"\n--- Stitching Round {round_num+1} ---")
        new_img_list = []
        for i in range(0, len(img_list)-1, 2):
            print(f"Stitching pair: {img_list[i]} + {img_list[i+1]}")
            stitched_name = iterative_superglue_stitch(img_list[i], img_list[i+1], input_dir, output_dir)
            new_img_list.append(stitched_name)

        # Handle odd image out
        if len(img_list) % 2 == 1:
            print(f"Carrying forward last image: {img_list[-1]}")
            new_img_list.append(img_list[-1])

        img_list = new_img_list
        round_num += 1

    print(f"\n✅ Final stitched image: {img_list[0]}")
    return img_list[0]

