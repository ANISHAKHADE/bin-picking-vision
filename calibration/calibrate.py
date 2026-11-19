"""
Camera calibration bonus exercise (OpenCV checkerboard method).

What to do before running this:
1. Print a checkerboard pattern (search "opencv checkerboard pattern pdf" — any 9x6 or 7x5
   INTERNAL-corner pattern works, just update CHECKERBOARD below to match). Stick it on
   something flat (a book, a clipboard).
2. Take 15-20 photos of it with your phone from different angles/distances/tilts — vary it a
   lot, this is what makes calibration well-conditioned. Keep the whole board in frame.
3. Put those photos in calibration/images/ (jpg or png).
4. Run: python calibration/calibrate.py

What this produces (in results/calibration/):
- camera_matrix.txt / dist_coeffs.txt — the actual calibration output
- undistorted_sample.png — a before/after pair on one of your images
- A printed reprojection error (lower = better calibration; under ~0.5 px is solid for a phone cam)

Why this matters for robot vision (so you can explain it in an interview): before you can map a
detected pixel to a real-world 3D point (needed for pose estimation / grasp planning), you need
the camera's intrinsic parameters (focal length, principal point, lens distortion). This script
does the intrinsic half of that; extrinsic calibration (camera pose relative to the robot base)
is the other half this does NOT cover.
"""

from pathlib import Path

import cv2
import numpy as np

# --- Config: update to match your printed checkerboard (count of INTERNAL corners) ---
CHECKERBOARD = (9, 6)  # (columns, rows) of internal corners — e.g. a 10x7 squares board
SQUARE_SIZE_MM = 25.0  # real-world size of one square edge; only affects scale, not error

IMAGES_DIR = Path(__file__).parent / "images"
OUT_DIR = Path(__file__).parent.parent / "results" / "calibration"


def main():
    image_paths = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpeg")))
    if len(image_paths) < 10:
        print(f"Found only {len(image_paths)} images in {IMAGES_DIR}. Aim for 15-20 before calibrating.")
        if len(image_paths) == 0:
            return

    # 3D points of a checkerboard are just a flat grid: (0,0,0), (1,0,0), ... scaled by square size
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []  # 3D points, one set per successfully-detected image
    imgpoints = []  # corresponding 2D corner locations
    img_shape = None
    used, skipped = 0, 0

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            skipped += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if not found:
            skipped += 1
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners_refined)
        used += 1

    print(f"Checkerboard detected in {used}/{len(image_paths)} images ({skipped} skipped — bad angle/blur/lighting).")
    if used < 8:
        print("Too few successful detections for a reliable calibration. Retake photos with the full board visible.")
        return

    reproj_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT_DIR / "camera_matrix.txt", camera_matrix)
    np.savetxt(OUT_DIR / "dist_coeffs.txt", dist_coeffs)

    print("\n--- Calibration result ---")
    print(f"Reprojection error: {reproj_error:.4f} px  (lower is better; <0.5 is solid for a phone camera)")
    print("Camera matrix (intrinsics):")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs.ravel())

    # Before/after undistortion on one sample image, saved side-by-side
    sample_path = image_paths[0]
    sample_img = cv2.imread(str(sample_path))
    h, w = sample_img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(sample_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    side_by_side = np.hstack([sample_img, undistorted])
    cv2.imwrite(str(OUT_DIR / "undistorted_sample.png"), side_by_side)

    print(f"\nSaved camera_matrix.txt, dist_coeffs.txt, and undistorted_sample.png to {OUT_DIR}")
    print("Copy the reprojection error and matrix into the README Results section.")


if __name__ == "__main__":
    main()
