import cv2
import numpy as np
import glob
import open3d as o3d

# ----------------------------------------------------------------------------
# 1. CAMERA CALIBRATION (UNCHANGED)
# ----------------------------------------------------------------------------
chessboard_size = (8, 5)  # 8×5 inner corners
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= 3  # scale each square by “3 units”

objpoints = []
imgpoints = []

calib_images = glob.glob(
    'C:/Users/chauh/Desktop/KCL_Subjects/Sensing_and_Perception/SAP_CW1_k24078085'
    '/SAP_CW1_k24078085/Dataset/chessboard/*.JPG'
)
for fname in calib_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if not ret:
        continue

    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_crit)

    objpoints.append(objp)
    imgpoints.append(corners_sub)
    cv2.drawChessboardCorners(img, chessboard_size, corners_sub, ret)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
print("Camera intrinsic matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.ravel())
print("Calibration reprojection error:\n", ret)

# ----------------------------------------------------------------------------
# 2. LOAD SfM IMAGES & EXTRACT SIFT FEATURES
# ----------------------------------------------------------------------------
image_paths = sorted(glob.glob(
    'C:/Users/chauh/Desktop/KCL_Subjects/Sensing_and_Perception'
    '/SAP_CW1_k24078085/SAP_CW1_k24078085/Dataset/sippu/*.JPG'
))
# Load and verify each image
images = []
for p in image_paths:
    img = cv2.imread(p)
    if img is None:
        print(f"Warning: could not load image at {p}")
        continue
    images.append(img)

N_images = len(images)
print(f"Number of valid SfM images: {N_images}")
if N_images < 2:
    raise RuntimeError("Need at least two images for SfM and geometric verification.")

# SIFT detector
sift = cv2.SIFT_create()
keypoints_list = []
descriptors_list = []
for idx, img in enumerate(images):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        # If no descriptors found, produce an empty array so indexing remains consistent
        kp = []
        desc = np.zeros((0, 128), dtype=np.float32)
    keypoints_list.append(kp)
    descriptors_list.append(desc)
    print(f"Image {idx}: detected {len(kp)} SIFT keypoints.")

# ----------------------------------------------------------------------------
# 3. BUILD VIEW‐GRAPH: PAIRWISE MATCHING + FUNDAMENTAL‐RANSAC
# ----------------------------------------------------------------------------
# view_graph_matches will store, for each pair (i,j) with i<j:
#   - 'corr_idx':  (M × 2) array of (kp_index_in_i, kp_index_in_j)
#   - 'inliers_i' and 'inliers_j':  (M × 2) arrays of undistorted image coords
#   - 'F':  the 3×3 fundamental matrix from RANSAC
view_graph_matches = {}

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
ratio_thresh = 0.75

def undistort_points(pts, K, dist_coeffs):
    """
    Undistort pixel points using cv2.undistortPoints with P=K,
    so the output is again in pixel coordinates but with distortion removed.
    """
    pts = pts.reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
    return und.reshape(-1, 2)

for i in range(N_images):
    for j in range(i + 1, N_images):
        desc_i = descriptors_list[i]
        desc_j = descriptors_list[j]
        if desc_i.shape[0] < 2 or desc_j.shape[0] < 2:
            # Not enough descriptors to match
            continue

        knn = bf.knnMatch(desc_i, desc_j, k=2)
        # Lowe’s ratio test
        good = []
        for m, n in knn:
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        if len(good) < 8:
            continue

        pts_i = np.float32([keypoints_list[i][m.queryIdx].pt for m in good])
        pts_j = np.float32([keypoints_list[j][m.trainIdx].pt for m in good])

        F_ij, mask_ij = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 3.0, 0.99)
        if F_ij is None:
            continue
        mask_ij = mask_ij.ravel().astype(bool)
        if np.count_nonzero(mask_ij) < 8:
            continue

        inl_i = pts_i[mask_ij]
        inl_j = pts_j[mask_ij]
        # Undistort those inliers
        und_i = undistort_points(inl_i, camera_matrix, dist_coeffs)
        und_j = undistort_points(inl_j, camera_matrix, dist_coeffs)

        corr_idx = np.array([[good[k].queryIdx, good[k].trainIdx]
                             for k in np.where(mask_ij)[0]])

        view_graph_matches[(i, j)] = {
            'corr_idx': corr_idx,       # shape (M_ij, 2)
            'inliers_i': und_i,         # shape (M_ij, 2)
            'inliers_j': und_j,         # shape (M_ij, 2)
            'F': F_ij                   # 3×3 fundamental matrix
        }
        print(f"Pair ({i},{j}): {len(und_i)} RANSAC inliers stored.")

# ----------------------------------------------------------------------------
# 4. GEOMETRIC VERIFICATION VIA CYCLE‐CONSISTENCY (TRIPLET CHECK)
# ----------------------------------------------------------------------------
def build_undistorted_lookup(i, vgm):
    """
    For image i, build an (N_kp_i × 2) array such that
    und_map[i][kp_index] = (x, y) of that keypoint in undistorted pixel coords,
    for any kp_index that appears in at least one pair (i, *). Others remain NaN.
    """
    num_kp_i = len(keypoints_list[i])
    und_map = np.full((num_kp_i, 2), np.nan, dtype=np.float32)
    for (a, b), data in vgm.items():
        if a == i:
            corr = data['corr_idx']    # shape (M, 2): [idx_in_a, idx_in_b]
            und_pts = data['inliers_i']  # shape (M, 2) for image a
            for row, (idx_a, _) in enumerate(corr):
                und_map[idx_a] = und_pts[row]
        elif b == i:
            corr = data['corr_idx']
            und_pts = data['inliers_j']  # shape (M, 2) for image b
            for row, (_, idx_b) in enumerate(corr):
                und_map[idx_b] = und_pts[row]
    return und_map

def build_triplet_tracks(i, j, k, vgm):
    """
    Return a list of (idx_i, idx_j, idx_k) such that:
      (idx_i → idx_j) is in edge (i,j)
      (idx_j → idx_k) is in edge (j,k)
      (idx_i → idx_k) is in edge (i,k)
    All indexes refer to keypoint‐indices in each image.
    """
    corr_ij = vgm[(i, j)]['corr_idx']
    corr_jk = vgm[(j, k)]['corr_idx']
    corr_ik = vgm[(i, k)]['corr_idx']

    match_i_j = {u_i: u_j for u_i, u_j in corr_ij}
    match_j_k = {u_j: u_k for u_j, u_k in corr_jk}
    match_i_k = {u_i: u_k for u_i, u_k in corr_ik}

    tracks = []
    for idx_i, idx_j in corr_ij:
        if idx_j in match_j_k:
            idx_k = match_j_k[idx_j]
            if (idx_i in match_i_k) and (match_i_k[idx_i] == idx_k):
                tracks.append((idx_i, idx_j, idx_k))
    return tracks

def is_cycle_consistent(i, j, k, idx_i, idx_j, idx_k, vgm, thr=1e-3):
    """
    Given a triplet of keypoint‐indices that appear matched in all three edges,
    check epipolar consistency for F_ij, F_jk, F_ik:
      x_i^T F_ij x_j ≈ 0,  x_j^T F_jk x_k ≈ 0,  x_i^T F_ik x_k ≈ 0
    where x_i, x_j, x_k are homogeneous undistorted pixel coordinates.
    """
    F_ij = vgm[(i, j)]['F']
    F_jk = vgm[(j, k)]['F']
    F_ik = vgm[(i, k)]['F']

    und_i = build_undistorted_lookup(i, vgm)
    und_j = build_undistorted_lookup(j, vgm)
    und_k = build_undistorted_lookup(k, vgm)

    x_i = np.array([*und_i[idx_i], 1.0])
    x_j = np.array([*und_j[idx_j], 1.0])
    x_k = np.array([*und_k[idx_k], 1.0])

    e_ij = float(x_i.T @ F_ij @ x_j)
    e_jk = float(x_j.T @ F_jk @ x_k)
    e_ik = float(x_i.T @ F_ik @ x_k)

    return (abs(e_ij) < thr) and (abs(e_jk) < thr) and (abs(e_ik) < thr)


# Iterate over all triplets (i,j,k) where i<j<k and all three edges exist
for i in range(N_images):
    for j in range(i + 1, N_images):
        for k in range(j + 1, N_images):
            if (i, j) not in view_graph_matches or \
               (j, k) not in view_graph_matches or \
               (i, k) not in view_graph_matches:
                continue

            tracks_ijk = build_triplet_tracks(i, j, k, view_graph_matches)
            if len(tracks_ijk) == 0:
                continue

            bad_triplets = []
            for (idx_i, idx_j, idx_k) in tracks_ijk:
                if not is_cycle_consistent(i, j, k, idx_i, idx_j, idx_k, view_graph_matches, thr=1e-3):
                    bad_triplets.append((idx_i, idx_j, idx_k))

            if not bad_triplets:
                continue

            # Prune each of the three edges by removing any correspondence
            # that appears in bad_triplets
            def prune_edge(i1, i2, bads):
                if (i1, i2) not in view_graph_matches:
                    return
                data = view_graph_matches[(i1, i2)]
                corr = data['corr_idx']      # shape (M, 2)
                inl_i1 = data['inliers_i']   # shape (M, 2)
                inl_i2 = data['inliers_j']   # shape (M, 2)
                F12 = data['F']

                keep_mask = np.ones(len(corr), dtype=bool)
                for (bi, bj, bk) in bads:
                    if (i1, i2) == (i, j):
                        rows = np.where((corr[:, 0] == bi) & (corr[:, 1] == bj))[0]
                    elif (i1, i2) == (j, k):
                        rows = np.where((corr[:, 0] == bj) & (corr[:, 1] == bk))[0]
                    elif (i1, i2) == (i, k):
                        rows = np.where((corr[:, 0] == bi) & (corr[:, 1] == bk))[0]
                    else:
                        rows = np.array([], dtype=int)
                    keep_mask[rows] = False

                if keep_mask.sum() < 8:
                    # If too few left, delete entire edge
                    del view_graph_matches[(i1, i2)]
                    return

                new_corr = corr[keep_mask]
                new_inl_i1 = inl_i1[keep_mask]
                new_inl_i2 = inl_i2[keep_mask]
                view_graph_matches[(i1, i2)] = {
                    'corr_idx': new_corr,
                    'inliers_i': new_inl_i1,
                    'inliers_j': new_inl_i2,
                    'F': F12
                }

            prune_edge(i, j, bad_triplets)
            prune_edge(j, k, bad_triplets)
            prune_edge(i, k, bad_triplets)

            print(f"Pruned {len(bad_triplets)} inconsistent tracks in triplet ({i},{j},{k}).")

# ----------------------------------------------------------------------------
# 5. (OPTIONAL) RE‐FIT FUNDAMENTAL MATRIX FOR EACH PRUNED EDGE
# ----------------------------------------------------------------------------
for (i, j), data in list(view_graph_matches.items()):
    corr = data['corr_idx']
    if corr.shape[0] < 8:
        del view_graph_matches[(i, j)]
        continue

    und_map_i = build_undistorted_lookup(i, view_graph_matches)
    und_map_j = build_undistorted_lookup(j, view_graph_matches)

    pts_i = np.array([und_map_i[idx_i] for idx_i, _ in corr], dtype=np.float32)
    pts_j = np.array([und_map_j[idx_j] for _, idx_j in corr], dtype=np.float32)

    F_new, mask_new = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 3.0, 0.99)
    if F_new is None:
        del view_graph_matches[(i, j)]
        continue

    mask_new = mask_new.ravel().astype(bool)
    if mask_new.sum() < 8:
        del view_graph_matches[(i, j)]
        continue

    view_graph_matches[(i, j)]['F'] = F_new
    view_graph_matches[(i, j)]['corr_idx'] = corr[mask_new]
    view_graph_matches[(i, j)]['inliers_i'] = pts_i[mask_new]
    view_graph_matches[(i, j)]['inliers_j'] = pts_j[mask_new]
    print(f"Re‐fitted F for edge ({i},{j}): {mask_new.sum()} inliers remain.")

# ----------------------------------------------------------------------------
# 6. RECOVER CAMERA POSES (ESSENTIAL → R, t) + BUILD GLOBAL POSE LIST
# ----------------------------------------------------------------------------
# Initialize R_world, t_world
R_world = [None] * N_images
t_world = [None] * N_images
R_world[0] = np.eye(3, dtype=np.float64)
t_world[0] = np.zeros((3, 1), dtype=np.float64)

# Use edge (0,1) if it exists to initialize camera 1
if (0, 1) in view_graph_matches:
    pts0 = view_graph_matches[(0, 1)]['inliers_i']
    pts1 = view_graph_matches[(0, 1)]['inliers_j']
    E01, _ = cv2.findEssentialMat(pts0, pts1, camera_matrix, cv2.RANSAC, 0.999, 1.0)
    _, R01, t01, _ = cv2.recoverPose(E01, pts0, pts1, camera_matrix)
    R_world[1] = R01.copy()
    t_world[1] = t01.copy()
    print("Initialized pose for camera 1 from (0,1).")
else:
    raise RuntimeError("Edge (0,1) not found after pruning. Cannot initialize second camera.")

# BFS‐style propagation through the view‐graph
visited = set([0, 1])
queue = [0, 1]

while queue:
    cur = queue.pop(0)
    for (i, j), data in list(view_graph_matches.items()):
        # Case: cur == i, solve for j
        if i == cur and j not in visited:
            pts_i = data['inliers_i']
            pts_j = data['inliers_j']
            Eij, _ = cv2.findEssentialMat(pts_i, pts_j, camera_matrix, cv2.RANSAC, 0.999, 1.0)
            _, Rrel, trel, _ = cv2.recoverPose(Eij, pts_i, pts_j, camera_matrix)
            R_world[j] = R_world[i] @ Rrel
            t_world[j] = R_world[i] @ trel + t_world[i]
            visited.add(j)
            queue.append(j)
            print(f"Solved pose for camera {j} from edge ({i},{j}).")

        # Case: cur == j, solve for i (need to invert relative pose)
        elif j == cur and i not in visited:
            pts_i = data['inliers_i']
            pts_j = data['inliers_j']
            # Note: recoverPose(Eji, pts_j, pts_i) → (Rji, tji) transforms i→j inverted
            Eji, _ = cv2.findEssentialMat(pts_j, pts_i, camera_matrix, cv2.RANSAC, 0.999, 1.0)
            _, Rji, tji, _ = cv2.recoverPose(Eji, pts_j, pts_i, camera_matrix)
            Rij = Rji.T
            tij = -Rij @ tji
            R_world[i] = R_world[j] @ Rij
            t_world[i] = R_world[j] @ tij + t_world[j]
            visited.add(i)
            queue.append(i)
            print(f"Solved pose for camera {i} from edge ({i},{j}).")

# ----------------------------------------------------------------------------
# 7. TRIANGULATE ALL INLIER CORRESPONDENCES (PAIRWISE) INTO 3D POINTS
# ----------------------------------------------------------------------------
point_cloud = np.zeros((0, 3), dtype=np.float32)

for (i, j), data in view_graph_matches.items():
    if i >= j:
        continue

    R_i = R_world[i]
    t_i = t_world[i]
    R_j = R_world[j]
    t_j = t_world[j]

    P_i = camera_matrix @ np.hstack((R_i, t_i))
    P_j = camera_matrix @ np.hstack((R_j, t_j))

    pts_i = data['inliers_i'].T   # shape (2, M)
    pts_j = data['inliers_j'].T   # shape (2, M)

    pts4D = cv2.triangulatePoints(P_i, P_j, pts_i, pts_j)  # (4, M)
    pts4D /= pts4D[3 : 4, :]
    pts3D = pts4D[:3, :].T  # shape (M, 3)

    point_cloud = np.vstack((point_cloud, pts3D))

print(f"Triangulated a total of {point_cloud.shape[0]} 3D points from all pairs.")

# ----------------------------------------------------------------------------
# 8. VISUALIZE CAMERA TRAJECTORY & POINT CLOUD
# ----------------------------------------------------------------------------
camera_centers = []
for R, t in zip(R_world, t_world):
    C = -R.T @ t   # camera center in world coords
    camera_centers.append(C.flatten())
camera_centers = np.array(camera_centers)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
           s=1, c='blue', label='3D Points')
ax.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
        c='red', marker='o', label='Camera Trajectory')
ax.set_title('Camera Trajectory & 3D Point Cloud')
ax.legend()
plt.show()

# ----------------------------------------------------------------------------
# 9. SAVE POINT CLOUD (WITH COLOR)
# ----------------------------------------------------------------------------
def extract_colors(image, points_2d):
    """
    Given a color image and an (N×2) array of pixel coords,
    return an (N×3) uint8 array of RGB colors sampled at those pixels.
    """
    colors = []
    h, w = image.shape[:2]
    for pt in points_2d:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            b, g, r = image[y, x]
            colors.append((r, g, b))
        else:
            colors.append((255, 255, 255))
    return np.array(colors, dtype=np.uint8)

# Use the first image’s inlier 2D positions (inliers_pts1 from the (0,1) edge)
if (0, 1) in view_graph_matches:
    pts1_2d = view_graph_matches[(0, 1)]['inliers_i']
    colors = extract_colors(images[1], pts1_2d)  # sample from image 1
else:
    colors = np.tile(np.array([[200, 200, 200]], dtype=np.uint8), (point_cloud.shape[0], 1))

def save_colored_point_cloud_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

output_ply = "reconstruction_colored.ply"
save_colored_point_cloud_ply(output_ply, point_cloud, colors)
print(f"[Info] Saved colored point cloud ({point_cloud.shape[0]} points) to {output_ply}")

# ----------------------------------------------------------------------------
# 10. MESHING (UNCHANGED)
# ----------------------------------------------------------------------------
pcd = o3d.io.read_point_cloud(output_ply)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
pcd.orient_normals_consistent_tangent_plane(k=100)

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
densities = np.asarray(densities)
density_threshold = np.percentile(densities, 10)
vertices = np.asarray(mesh.vertices)
mesh_filtered = mesh.select_by_index(np.where(densities > density_threshold)[0])

o3d.io.write_triangle_mesh("mesh_poisson_trimmed.ply", mesh_filtered)
o3d.visualization.draw_geometries([mesh_filtered])
