# === 0. 套件 ===
import os, glob, time
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
from typing import Dict, List
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from pycpd import RigidRegistration  # pip install pycpd
import math
DATASETS: List[Dict] = [
{
        "name": "red01",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red01",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red01",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red01/Image_20240321100710525.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321100710525.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/red01_combined.xlsx",
    },
    {
        "name": "red02",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red02",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red02",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red02/Image_20240321104026349.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321104026349.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/red02_combined.xlsx",
    },
    {
        "name": "red03",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red03",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red03",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red03/Image_20240321105545542.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321105545542.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/red03_combined.xlsx",
    },
    {
        "name": "red04",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red04",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red04",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red04/Image_20240321111055812.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321111055812.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/red04_combined.xlsx",
    },
    {
        "name": "red05",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red05",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red05",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red05/Image_20240321112738655.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321112738655.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/red05_combined.xlsx",
    },
    {
        "name": "red06",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red06",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red06",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red06/Image_20240321114408009.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321114408009.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/red06_combined.xlsx",
    },
    {
        "name": "white01",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white01",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white01",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white01/Image_20240321143412883.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321143412883.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/white01_combined.xlsx",
    },
    {
        "name": "white02",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white02",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white02",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white02/Image_20240321145312162.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321145312162.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/white02_combined.xlsx",
    },
    {
        "name": "white03",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white03",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white03",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white03/Image_20240321151040617.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321151040617.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/white03_combined.xlsx",
    },
    {
        "name": "white04",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white04",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white04",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white04/Image_20240321152853920.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321152853920.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/white04_combined.xlsx",
    },
    {
        "name": "white05",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white05",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white05",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white05/Image_20240321154611642.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321154611642.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/white05_combined.xlsx",
    },
    {
        "name": "white06",
        "csv_dir":  r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white06",
        "img_dir":  r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white06",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white06/Image_20240321160945507.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321160945507.bmp",
        "final_xlsx": r"/media/dc0206/Crucial X6/GMM20/white06_combined.xlsx",
    },

    # --- 新資料集請直接在此處繼續加入 ------------------------------------
]
# ---------------------------
# 1. 補點函式
# ---------------------------
def ensure_two_points_in_circle_all(filtered_points, radius: float = 10.0):
    """確保每個圓形區域內至少兩點，如不足則補點"""
    if len(filtered_points) < 1:
        return filtered_points, []

    new_points = []
    for i, center in enumerate(filtered_points):
        dists = np.linalg.norm(filtered_points - center, axis=1)
        in_circle_indices = np.where(dists <= radius)[0]
        if len(in_circle_indices) < 2:
            if len(filtered_points) == 1:
                rand_dir = np.random.randn(2)
                rand_dir /= (np.linalg.norm(rand_dir) + 1e-9)
                new_pt = center + rand_dir * (radius * 0.5)
                new_points.append(new_pt)
            else:
                sorted_idx = np.argsort(dists)
                nearest_idx = sorted_idx[1] if sorted_idx[0] == i else sorted_idx[0]
                nearest_point = filtered_points[nearest_idx]
                direction_vec = center - nearest_point
                dist_np = np.linalg.norm(direction_vec)
                if dist_np < 1e-9:
                    direction_vec = np.random.randn(2)
                    dist_np = np.linalg.norm(direction_vec)
                dir_unit = direction_vec / dist_np
                new_pt = center + dir_unit * (radius * 0.8)
                new_points.append(new_pt)

    if len(new_points) > 0:
        new_points = np.array(new_points)
        updated_points = np.vstack([filtered_points, new_points])
    else:
        updated_points = filtered_points
    return updated_points, new_points

def ensure_two_points_in_circle_iterative(points, radius: float = 10.0, max_iter: int = 100):
    updated_points = points.copy()
    for _ in range(max_iter):
        updated_points, new_points = ensure_two_points_in_circle_all(updated_points, radius=radius)
        if len(new_points) == 0:
            break
    return updated_points

def fill_line_gaps(points, *, max_distance: float = 3.0, max_iter: int = 5):
    pts = points.copy()
    for _ in range(max_iter):
        nbrs = NearestNeighbors(n_neighbors=2).fit(pts)
        distances, indices = nbrs.kneighbors(pts)
        new_pts = []
        for i in range(len(pts)):
            d = distances[i, 1]
            if d > max_distance:
                p1, p2 = pts[i], pts[indices[i, 1]]
                num_insert = int(d // max_distance)
                for n in range(1, num_insert + 1):
                    new_pts.append(p1 + (p2 - p1) * n / (num_insert + 1))
        if not new_pts:
            break
        pts = np.vstack([pts, np.array(new_pts)])
    return pts
def preprocess_fill_gap(cur,
                        radius: float = 0.5,
                        max_iter_circle: int = 300,
                        max_distance: float = 2.0,
                        max_iter_fill: int = 500):
    """
    對曲線點雲做前處理：去雜訊、確保兩點在圓內、填補斷點。

    參數：
    - cur: 原始點雲資料
    - radius: ensure_two_points_in_circle_iterative 的半徑參數，預設 0.5
    - max_iter_circle: ensure_two_points_in_circle_iterative 最大迭代次數，預設 300
    - max_distance: fill_line_gaps 的最大距離參數，預設 2.0
    - max_iter_fill: fill_line_gaps 最大迭代次數，預設 500

    回傳：
    - 經過三步驟處理後的點雲 cur
    """
    # 1. 加權去雜訊，取回第一組結果
    cur = filter_noise_points_weighted(cur)[0]

    # 2. 確保至少有兩點在指定半徑的圓內
    cur = ensure_two_points_in_circle_iterative(
        cur,
        radius=radius,
        max_iter=max_iter_circle
    )

    # 3. 填補線段斷點
    cur = fill_line_gaps(
        cur,
        max_distance=max_distance,
        max_iter=max_iter_fill
    )

    return cur

# ---------------------------
# 2. 雜訊過濾與輔助函式
# ---------------------------
def filter_noise_points_weighted(points, eps: float = 3, min_samples: int = 2,
                                 min_cluster_size: int = 55, return_mask: bool = False):
    if len(points) == 0:
        return np.array([]), np.array([]) if not return_mask else (np.array([]), np.array([]), None)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    large_clusters = set(unique_labels[counts >= min_cluster_size])
    mask = np.isin(labels, list(large_clusters))
    filtered_points = points[mask]
    return (filtered_points, mask) if not return_mask else (filtered_points, mask, mask)

def compute_local_density(points, k: int = 18):
    if len(points) == 0:
        return np.array([])
    actual_k = min(k + 1, len(points))
    nbrs = NearestNeighbors(n_neighbors=actual_k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    avg_distance = np.mean(distances[:, 1:], axis=1)
    density = 1.0 / (avg_distance + 1e-6)
    return density

def estimate_rigid_transform(src, dst):
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_dst - R @ centroid_src
    return R, t

def pca_coarse_alignment(src: np.ndarray, dst: np.ndarray):

    # 1. 計算質心
    src_cent = src.mean(axis=0)
    dst_cent = dst.mean(axis=0)
    
    # 2. 中心化
    src_c = src - src_cent    # shape = (N,2)
    dst_c = dst - dst_cent    # shape = (M,2)

    # 3. 分別計算共變異數矩陣
    Cov_src = src_c.T @ src_c   # (2,2)
    Cov_dst = dst_c.T @ dst_c   # (2,2)

    # 4. SVD 分解得到主方向
    U_src, _, _ = np.linalg.svd(Cov_src)
    U_dst, _, _ = np.linalg.svd(Cov_dst)

    # 5. 主軸對齊：讓 src 的主方向轉到 dst 的主方向
    R = U_dst @ U_src.T

    # 6. 處理鏡像情況
    if np.linalg.det(R) < 0:
        # 強制讓 R 變成純旋轉
        U_dst[:,1] *= -1
        R = U_dst @ U_src.T

    # 7. 平移對齊質心
    t = dst_cent - R @ src_cent

    return R, t

# ---------------------------
# 3. ICP / CPD / Normal‑ICP
# ---------------------------
def icp_with_weights(source_points, target_points, *, max_iterations: int = 500, tolerance: float = 1e-9):#-20
    # 計算點權重
    dens = compute_local_density(source_points, k=25)
    dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-9)
    weights = dens
    R_total = np.eye(2)
    t_total = np.zeros(2)
    prev_error = float("inf")
    for iteration in range(max_iterations):
        transformed_source = (R_total @ source_points.T).T + t_total
        nbrs = NearestNeighbors(n_neighbors=1).fit(target_points)
        distances, indices = nbrs.kneighbors(transformed_source)
        closest_points = target_points[indices.flatten()]
        error = np.sum(weights * (distances.flatten() ** 2)) / (np.sum(weights) + 1e-9)
        if abs(prev_error - error) < tolerance and error < 1:
            break
        prev_error = error
        src_centroid = np.average(transformed_source, axis=0, weights=weights)
        tgt_centroid = np.average(closest_points, axis=0, weights=weights)
        src_cent = transformed_source - src_centroid
        tgt_cent = closest_points - tgt_centroid
        H = (weights[:, None] * src_cent).T @ tgt_cent
        U, _, Vt = np.linalg.svd(H)
        R_delta = Vt.T @ U.T
        if np.linalg.det(R_delta) < 0:
            Vt[1, :] *= -1
            R_delta = Vt.T @ U.T
        t_delta = tgt_centroid - R_delta @ src_centroid
        R_total = R_delta @ R_total
        t_total = R_delta @ t_total + t_delta
    return R_total, t_total, prev_error

def cpd_rigid_registration(source_points, target_points, *, max_iterations: int = 50, tolerance: float = 1e-3):

    reg = RigidRegistration(X=target_points, Y=source_points,
                            max_iterations=max_iterations, tolerance=tolerance)
    Y_reg, (s, R, t) = reg.register()
    return s, R, t, Y_reg

def compute_normals(points, k: int = 40):
    if len(points) < 3:
        return np.zeros_like(points)
    actual_k = min(k, len(points) - 1)
    nbrs = NearestNeighbors(n_neighbors=actual_k).fit(points)
    _, indices = nbrs.kneighbors(points)
    normals = np.zeros_like(points)
    for i, neighbors in enumerate(indices):
        cov_matrix = np.cov(points[neighbors].T)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        normals[i] = eigvecs[:, 0]
    return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

def normal_icp(source_points, target_points, *, alpha: float = 0.1, gamma: float = 0.8,
               max_iterations: int = 1000, tolerance: float = 1e-9):
    dens = compute_local_density(source_points, k=25)
    dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-9)
    weights = dens
    source_normals = compute_normals(source_points)
    target_normals = compute_normals(target_points)
    R_total = np.eye(2)
    t_total = np.zeros(2)
    prev_error = float("inf")
    for iteration in range(max_iterations):
        transformed_source = (R_total @ source_points.T).T + t_total
        nbrs = NearestNeighbors(n_neighbors=1).fit(target_points)
        distances, indices = nbrs.kneighbors(transformed_source)
        indices = np.clip(indices.flatten(), 0, len(target_normals) - 1)
        closest_points = target_points[indices]
        closest_normals = target_normals[indices]
        normal_dot = np.einsum("ij,ij->i", source_normals, closest_normals)
        normal_errors = np.arccos(np.clip(normal_dot, -1, 1))
        weighted_normals = np.exp(-normal_errors / (2 * alpha)) ** gamma
        combined_weights = weights * distances.flatten() * weighted_normals
        error = np.sum(combined_weights * (distances.flatten() ** 2)) / (np.sum(combined_weights) + 1e-9)
        if abs(prev_error - error) < tolerance and error < 1:
            break
        prev_error = error
        src_centroid = np.average(transformed_source, axis=0, weights=combined_weights)
        tgt_centroid = np.average(closest_points, axis=0, weights=combined_weights)
        src_cent = transformed_source - src_centroid
        tgt_cent = closest_points - tgt_centroid
        H = (combined_weights[:, None] * src_cent).T @ tgt_cent
        U, _, Vt = np.linalg.svd(H)
        R_delta = Vt.T @ U.T
        if np.linalg.det(R_delta) < 0:
            Vt[1, :] *= -1
            R_delta = Vt.T @ U.T
        t_delta = tgt_centroid - R_delta @ src_centroid
        R_total = R_delta @ R_total
        t_total = R_delta @ t_total + t_delta
    return R_total, t_total, prev_error
# ---------------------------
# 4. 角度估計建立或載入參考點與角度
# ---------------------------

def estimate_initial_rotation(theta_src: np.ndarray, theta_ref: np.ndarray) -> float:
    bin_edges = np.arange(0, 361)
    hist_src, _ = np.histogram(theta_src % 360, bins=bin_edges)
    hist_ref, _ = np.histogram(theta_ref % 360, bins=bin_edges)
    fft_src = np.fft.rfft(hist_src)
    fft_ref = np.fft.rfft(hist_ref)
    corr = np.fft.irfft(fft_src * np.conj(fft_ref))
    phi0 = int(np.argmax(corr))
    rot_init = (phi0 + 180) % 360 - 180
    return float(math.radians(rot_init))

def match_template(cfg):
    """
    使用模版匹配（Template Matching）方法，在指定的第一張影像中搜尋並定位模板影像的位置。

    參數:
        cfg (dict): 配置字典，應包含以下鍵：
            - 'template_img': 模板影像路徑 (絕對或相對)
            - 'img_dir': 圖片目錄路徑
            - 'first_img': 目標第一張影像檔名

    回傳:
        tuple: (top_left, bottom_right)，分別為模板在第一張影像中的左上及右下座標
    """
    # 1. 以灰階方式讀取模板影像
    template_img = cv2.imread(cfg['template_img'], cv2.IMREAD_GRAYSCALE)
    if template_img is None:
        raise FileNotFoundError(f"找不到模板影像：{cfg['template_img']}")

    # 2. 組合路徑並讀取第一張影像
    first_img_path = os.path.join(cfg['img_dir'], cfg['first_img'])
    first_image = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise FileNotFoundError(f"找不到影像：{first_img_path}")

    # 3. 執行模板匹配（歸一化相關係數法）
    res_match = cv2.matchTemplate(first_image, template_img, cv2.TM_CCOEFF_NORMED)
    # 4. 取得最大值位置 (最佳匹配)
    _, _, _, top_left = cv2.minMaxLoc(res_match)

    # 5. 計算模板邊長，並獲得右下座標
    w, h = template_img.shape[1], template_img.shape[0]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    first_img_name = cfg['first_img']
    template_img = cv2.imread(cfg['template_img'], cv2.IMREAD_GRAYSCALE)
    return w, h ,first_img_name, template_img

def initial_rotation_align(cur,
                           cur_angle: float,
                           ref_angles: list,
                           region_center: np.ndarray,
                           angle_threshold: float = 5.0):
    """
    根據目前角度與參考角度估計初始旋轉，並對點雲做旋轉對齊。

    參數：
    - cur: np.ndarray，形狀 (N,2)，待對齊的點雲
    - cur_angle: float，當前曲線或點雲的角度（單位：度）
    - ref_angles: list of float，參考角度列表（單位：度）
    - region_center: np.ndarray，形狀 (2,)，旋轉中心座標
    - angle_threshold: float，旋轉角度門檻（單位：度），只有當絕對值大於此值才做旋轉

    回傳：
    - cur_aligned: np.ndarray，旋轉對齊後的點雲
    - rot0: float，估計出的初始旋轉角度
    """
    # 1. 估計初始旋轉角度（度）
    rot0 = estimate_initial_rotation(cur_angle, ref_angles)
    theta = math.radians(rot0)

    # 2. 若角度超過門檻 (threshold)，才進行旋轉
    if abs(theta) > angle_threshold:
        # 建立旋轉矩陣
        R0 = np.array([[ math.cos(theta), -math.sin(theta)],
                       [ math.sin(theta),  math.cos(theta)]])
        # 以 region_center 為旋轉中心做對齊
        cur = (R0 @ (cur - region_center).T).T + region_center

    return cur, rot0

def get_cur_and_angles(idx, imgs, csvs, template_img, w, h, ref):
    """
    從 imgs[idx] 和 csvs[idx] 取出 ROI 裡的點雲 cur 和對應角度 cur_ang。
    若篩選後 cur<3 或 ref<3，回傳 (None, None)。
    """
    img_path = imgs[idx]
    img_name = os.path.basename(img_path)
    # 模板匹配取左上角 tl
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(im, template_img, cv2.TM_CCOEFF_NORMED)
    _, _, _, tl = cv2.minMaxLoc(res)
    x0, y0 = tl
    region_center = np.array([tl[0] + w/2.0, tl[1] + h/2.0])
    # 讀 CSV、取點雲和角度
    dfc = pd.read_csv(csvs[idx])
    pts = dfc[['PosX', 'PosY']].values
    angles = dfc['GradientAngle'].values

    # 篩選 ROI 內
    cond = (
        (pts[:,0] >= x0) & (pts[:,0] <= x0 + w) &
        (pts[:,1] >= y0) & (pts[:,1] <= y0 + h)
    )
    cur = pts[cond]
    cur_ang = angles[cond]

    # 點太少就跳過
    if len(cur) < 3 or len(ref) < 3:
        return None, None

    return cur, cur_ang, region_center, img_name

def append_alignment_entry(results,
                           image_file: str,
                           R_total: np.ndarray,
                           t_total: np.ndarray,
                           inc: np.ndarray,
                           disp: np.ndarray):
    
    results.append({
        "image_file":            image_file,
        "rotation_matrix":       R_total.tolist(),
        "translation_vector":    t_total.tolist(),
        "incremental_displacement":   inc.tolist(),
        "incremental_disp_norm":      float(np.linalg.norm(inc)),
        "displacement":                disp.tolist(),
        "displacement_norm":           float(np.linalg.norm(disp)),
        # 如果要回傳 alignment_error，再把下面這行打開即可：
        #"alignment_error":         float(err_f),
    })

def process_all_images(
                       cfg: Dict,
                       imgs,
                       csvs,
                       template_img=None,
                       ref=None,
                       ref_angles=None,
                       *,
                       do_template: bool = True,
                       do_preprocess: bool = True,
                       do_rotation: bool = True,
                       do_coarse: bool = True,
                       do_fine_icp: bool = True,
                       do_fine_cpd: bool = True):
    """
    如果 do_template=True，就從 template_img.shape 取得 w,h；
    否則以影像全圖當 ROI。
    """
    results = []
    icp_count = cpd_count = 0
    prev_inc = np.zeros(2)
    template_img   = cv2.imread(cfg["template_img"], cv2.IMREAD_GRAYSCALE)
    if template_img is None:
        raise FileNotFoundError(f"找不到模板：{'template_img'}")
    # 只有在模板對齊時才需要 w,h
    if do_template:
        h_tpl, w_tpl = template_img.shape  # 取得模板高、寬

    for idx in range(1, len(imgs)):
        img_path = imgs[idx]
        img_name = os.path.basename(img_path)
        im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue

        # 1. 決定 ROI
        if do_template:
            # 用模板匹配計算左上角 tl
            res = cv2.matchTemplate(im, template_img, cv2.TM_CCOEFF_NORMED)
            _, _, _, tl = cv2.minMaxLoc(res)
            x0, y0 = tl
            region_center = np.array([x0 + w_tpl/2, y0 + h_tpl/2])

            # 篩選 ROI 內的點
            df = pd.read_csv(csvs[idx])
            pts   = df[['PosX','PosY']].values
            ang   = df['GradientAngle'].values
            cond  = (
                (pts[:,0] >= x0) & (pts[:,0] <= x0 + w_tpl) &
                (pts[:,1] >= y0) & (pts[:,1] <= y0 + h_tpl)
            )
            cur, cur_ang = pts[cond], ang[cond]

        else:
            # 全圖當 ROI
            h_i, w_i = im.shape
            region_center = np.array([w_i/2, h_i/2])
            df = pd.read_csv(csvs[idx])
            cur     = df[['PosX','PosY']].values
            cur_ang = df['GradientAngle'].values

        if len(cur) < 3 or len(ref) < 3:
            continue
        # 1. 去噪補點
        if do_preprocess:
            cur = preprocess_fill_gap(cur, radius=0.5,
                                      max_iter_circle=300,
                                      max_distance=2.0,
                                      max_iter_fill=500)

        # 2. 初始旋轉校正
        if do_rotation:
            cur, _ = initial_rotation_align(
                cur=cur,
                cur_angle=cur_ang,
                ref_angles=ref_angles,
                region_center=region_center,
                angle_threshold=5.0
            )

        # 3. 粗對齊 (PCA)
        if do_coarse:
            R_coarse, t_coarse = pca_coarse_alignment(cur, ref)
        else:
            R_coarse, t_coarse = np.eye(2), np.zeros(2)
        cur_coarse = (R_coarse @ cur.T).T + t_coarse

        # 4. 精細對齊(細分)
        R_f, t_f, err_fine = np.eye(2), np.zeros(2), None

        # 4.1 如果同時打開兩個，則比較誤差選最小
        if do_fine_icp and do_fine_cpd:
            # ICP
            R_p, t_p, e_p = normal_icp(cur_coarse, ref)

            # CPD + Normal‑ICP
            try:
                _, R_cpd, t_cpd, _ = cpd_rigid_registration(
                    cur_coarse, ref,
                    max_iterations=10,
                    tolerance=1e-3
                )
            except:
                R_cpd, t_cpd = np.eye(2), np.zeros(2)
            cur_cpd = (R_cpd @ cur_coarse.T).T + t_cpd
            R_n, t_n, e_n = normal_icp(cur_cpd, ref)

            if e_p < e_n:
                icp_count += 1
                R_f, t_f, err_fine = R_p, t_p, e_p
            else:
                cpd_count += 1
                R_f = R_n @ R_cpd
                t_f = R_n @ t_cpd + t_n
                err_fine = e_n

        # 4.2 只跑 ICP
        elif do_fine_icp:
            R_p, t_p, e_p = normal_icp(cur_coarse, ref)
            icp_count += 1
            R_f, t_f, err_fine = R_p, t_p, e_p

        # 4.3 只跑 CPD+Normal‑ICP 10 -3
        elif do_fine_cpd:
            try:
                _, R_cpd, t_cpd, _ = cpd_rigid_registration(
                    cur_coarse, ref,
                    max_iterations=10,
                    tolerance=1e-1
                )
            except:
                R_cpd, t_cpd = np.eye(2), np.zeros(2)
            cur_cpd = (R_cpd @ cur_coarse.T).T + t_cpd
            R_n, t_n, e_n = normal_icp(cur_cpd, ref)
            cpd_count += 1
            R_f = R_n @ R_cpd
            t_f = R_n @ t_cpd + t_n
            err_fine = e_n

        # 5. 合併全域變換
        R_total = R_f @ R_coarse
        t_total = R_f @ t_coarse + t_f
        cur = (R_total @ cur.T).T + t_total

        # 6. 計算位移增量
        transformed_center = R_total @ region_center + t_total
        inc = transformed_center - region_center
        disp = inc - prev_inc
        prev_inc = inc.copy()

        # 7. 收集結果
        # 收集結果
        append_alignment_entry(results,image_file=img_name,
                            R_total=R_total, t_total=t_total,
                            inc=inc, disp=disp)

    return results, icp_count, cpd_count

def get_or_build_reference(
    name: str,
    cfg: Dict,
    *,
    do_template: bool = True,
    do_gap_fill: bool = True,
):
    """
    取得或重建參考點 (ref_points) 與角度 (ref_angles)。

    參數：
    - name:       參考檔名識別用字串
    - cfg:        配置字典，需包含：
                  'template_csv', 'img_dir', 'first_img',
                  'template_img', 'csv_dir'
    - do_template:  是否執行模板匹配；False 時使用整張影像作為 ROI
    - do_gap_fill:  是否執行「補齊斷點」等後處理步驟

    回傳：
    - ref0:  np.ndarray, shape=(N,2)，篩選後並（可選）補點的參考點
    - ang0:  np.ndarray, shape=(N,)，對應的角度
    """
    pt_path = f"ref_points_{name}.npy"
    ang_path = f"ref_angles_{name}.npy"
    if os.path.exists(pt_path) and os.path.exists(ang_path):
        return np.load(pt_path), np.load(ang_path)

    # 1. 讀 CSV 與影像
    tpl_df = pd.read_csv(cfg["template_csv"])
    img0 = cv2.imread(
        os.path.join(cfg["img_dir"], cfg["first_img"]),
        cv2.IMREAD_GRAYSCALE
    )
    df0 = pd.read_csv(
        os.path.join(cfg["csv_dir"], cfg["first_img"].replace('.bmp', '.csv'))
    )
    pts0 = df0[['PosX','PosY']].values.astype(np.float32)
    ang0 = df0['GradientAngle'].values.astype(np.float32)

    # 2. ROI 範圍：模板匹配或整張影像
    if do_template:
        tpl_img = cv2.imread(cfg["template_img"], cv2.IMREAD_GRAYSCALE)
        _, _, _, tl = cv2.minMaxLoc(
            cv2.matchTemplate(img0, tpl_img, cv2.TM_CCOEFF_NORMED)
        )
        br = (tl[0] + tpl_img.shape[1], tl[1] + tpl_img.shape[0])
        x0, y0 = tl
        x1, y1 = br
    else:
        # 全圖範圍
        h_img, w_img = img0.shape
        x0, y0 = 0, 0
        x1, y1 = w_img, h_img

    # 3. 篩選 ROI 內的點
    mask0 = (
        (pts0[:,0] >= x0) & (pts0[:,0] <= x1) &
        (pts0[:,1] >= y0) & (pts0[:,1] <= y1)
    )
    ref0 = pts0[mask0]
    ang0 = ang0[mask0]

    # 4. 去雜訊 (固定執行)
    ref0 = filter_noise_points_weighted(ref0)[0]
    ang0 = ang0[:len(ref0)]

    # 5. 補點 / 補 gap（可開關）
    if do_gap_fill:
        # 確保兩點在圓內
        ref0 = ensure_two_points_in_circle_iterative(
            ref0, radius=0.5, max_iter=300
        )
        ang0 = ang0[:len(ref0)]
        # 填補斷點
        ref0 = fill_line_gaps(ref0, max_distance=2.0, max_iter=500)
        # 補齊角度長度
        if len(ang0) < len(ref0):
            ang0 = np.pad(ang0, (0, len(ref0) - len(ang0)), 'edge')

    # 6. 儲存並回傳
    np.save(pt_path, ref0)
    np.save(ang_path, ang0)
    return ref0, ang0

# ---------------------------
# 主流程
# ---------------------------
def process_dataset(cfg: Dict):
    # 1. 建構參考點與角度
    ref_points, ref_angles = get_or_build_reference(
        cfg['name'], cfg,
        do_template=False,
        do_gap_fill=True
    )
    ref = ref_points.copy()
    # 2. 第一張影像檔名 & 預先讀好模板影像
    first_img_name = cfg['first_img']

    # 3. 初始化 results（放入第一筆單位對齊）
    results = []
    append_alignment_entry(
                            results, image_file=first_img_name,R_total=np.eye(2),t_total=np.zeros(2),inc=np.zeros(2),disp=np.zeros(2),
                                                        #alignment_error=0.0
                                )
    # 4. 取得所有影像與 CSV 路徑
    imgs = sorted(glob.glob(os.path.join(cfg['img_dir'], '*.bmp')))
    csvs = sorted(glob.glob(os.path.join(cfg['csv_dir'], '*.csv')))
    # 5. 呼叫 process_all_images（關鍵字參數都放後面）
    more_results, icp_n, cpd_n = process_all_images(
        cfg,
        imgs,
        csvs,
        template_img=None,
        ref=ref,
        ref_angles=ref_angles,
        do_template = True,
        do_preprocess = True,
        do_rotation = True,
        do_coarse = True,
        do_fine_icp = True,
        do_fine_cpd = True
    )
    # 6. 合併第一筆與後續結果
    results.extend(more_results)

    # 7. 印出統計並存檔
    print(f"ICP 被選 {icp_n} 次，CPD+Normal‑ICP 被選 {cpd_n} 次")
    os.makedirs(os.path.dirname(cfg['final_xlsx']), exist_ok=True)
    pd.DataFrame(results).to_excel(cfg['final_xlsx'], index=False)

def main():
    import time
    for cfg in DATASETS:
        try:
            start = time.time()
            print(f"Processing [{cfg['name']}]")
            process_dataset(cfg)
            ##end = time.time()
            print(f"✅ [{cfg['name']}] 完成，檔案：{cfg['final_xlsx']}，總耗時 {(time.time()-start):.1f}s")
        except Exception as e:
            print(f"[{cfg['name']}] 失敗：{e}")

if __name__ == "__main__":
    main()
