#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from pycpd import RigidRegistration

# -------------------- 1. 參數設定 --------------------
SAVE_VIS = True  # 是否輸出對齊前後可視化圖

DATASETS = {
    "red01": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red01",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red01",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red01/Image_20240321100710525.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321100710525.bmp"
    },
    "red02": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red02",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red02",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red02/Image_20240321104026349.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321104026349.bmp"
    },
    "red03": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red03",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red03",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red03/Image_20240321105545542.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321105545542.bmp"
    },
    "red04": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red04",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red04",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red04/Image_20240321111055812.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321111055812.bmp"
    },
    "red05": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red05",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red05",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red05/Image_20240321112738655.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321112738655.bmp"
    },
    "red06": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red06",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/red06",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/red06/Image_20240321114408009.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321114408009.bmp"
    },
    "white01": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white01",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white01",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white01/Image_20240321143412883.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321143412883.bmp"
    },   
    "white02": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white02",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white02",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white02/Image_20240321145312162.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321145312162.bmp"
    },   
    "white03": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white03",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white03",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white03/Image_20240321151040617.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321151040617.bmp"
    },   
    "white04": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white04",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white04",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white04/Image_20240321152853920.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321152853920.bmp"
    },   
    "white05": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white05",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white05",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white05/Image_20240321154611642.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321154611642.bmp"
    },   
    "white06": {
        "csv_dir": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white06",
        "img_dir": r"/media/dc0206/Crucial X6/GMM20/20240329_DATA2/20240329_DATA/NTHU_5x/white06",
        "template_csv": r"/media/dc0206/Crucial X6/GMM20/EdgeFinderResult/white06/Image_20240321160945507.csv",
        "template_img": r"/media/dc0206/Crucial X6/GMM20/crop_w5.bmp",
        "first_img": "Image_20240321160945507.bmp"
    }
}
# ---------------------------
"""
red01: Image_20240321100710525
red02: Image_20240321104026349
red03: Image_20240321105545542
red04: Image_20240321111055812
red05: Image_20240321112738655
red06: Image_20240321114408009
red07: Image_20240321132009094
white01: Image_20240321143412883
white02: Image_20240321145312162
white03: Image_20240321151040617
white04: Image_20240321152853920
white05: Image_20240321154611642
white06: Image_20240321160945507
white07: Image_20240321162726016
"""
parser = argparse.ArgumentParser(description="Rotation Robustness Test")
parser.add_argument('--set', default='red01', help='Dataset key')
args, _ = parser.parse_known_args()
if args.set not in DATASETS:
    raise ValueError(f"未知資料集 {args.set}")
P = DATASETS[args.set]
CSV_DIR = P['csv_dir']
IMG_DIR = P['img_dir']
TEMPLATE_CSV = P['template_csv']
TEMPLATE_IMG = P['template_img']
FIRST_IMG_NAME = P['first_img']
# 將可視化圖存到指定路徑下的資料集資料夾
BASE_VIS_DIR = r"/media/dc0206/Crucial X6/GMM20/r_test"
VIS_DIR = os.path.join(BASE_VIS_DIR, args.set)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# -------------------- 2. 工具函式 --------------------
def ensure_two_points_in_circle_all(points, radius=10.0):
    if len(points) < 1:
        return points, []
    new_pts = []
    for i, c in enumerate(points):
        d = np.linalg.norm(points - c, axis=1)
        idx_in = np.where(d <= radius)[0]
        if len(idx_in) < 2:
            if len(points) == 1:
                v = np.random.randn(2)
                v /= np.linalg.norm(v) + 1e-9
                new_pts.append(c + v * radius * 0.5)
            else:
                order = np.argsort(d)
                j = order[1] if order[0] == i else order[0]
                vec = c - points[j]
                norm = np.linalg.norm(vec)
                if norm < 1e-9:
                    vec = np.random.randn(2)
                    norm = np.linalg.norm(vec)
                new_pts.append(c + vec / norm * radius * 0.8)
    if new_pts:
        new_pts = np.array(new_pts)
        return np.vstack([points, new_pts]), new_pts
    return points, []

def ensure_two_points_in_circle_iterative(points, radius=10.0, max_iter=100):
    pts = points.copy()
    for _ in range(max_iter):
        pts, added = ensure_two_points_in_circle_all(pts, radius)
        if len(added) == 0:
            break
    return pts

def filter_noise_points_weighted(points, eps=3, min_samples=2, min_cluster_size=55):
    if len(points) == 0:
        return np.empty((0,2))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    lbl = db.labels_
    ul, cnt = np.unique(lbl[lbl!=-1], return_counts=True)
    keep = set(ul[cnt >= min_cluster_size])
    mask = np.isin(lbl, list(keep))
    return points[mask]

def compute_local_density(points, k=25):
    if len(points) == 0:
        return np.array([])
    nn = min(k+1, len(points))
    nbrs = NearestNeighbors(n_neighbors=nn).fit(points)
    dist, _ = nbrs.kneighbors(points)
    return 1.0 / (np.mean(dist[:,1:], axis=1) + 1e-6)

def estimate_rigid_transform(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    H = (src-cs).T @ (dst-cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T
    t = cd - R @ cs
    return R, t

def usac_rigid_transform(src, dst, num_iterations=100, inlier_threshold=40.0, min_inliers=20):
    best_count = 0
    best_R, best_t = np.eye(2), np.zeros(2)
    N = len(src)
    for _ in range(num_iterations):
        if N < 8:
            break
        idx = np.random.choice(N, 8, replace=False)
        R_c, t_c = estimate_rigid_transform(src[idx], dst[idx])
        transformed = (R_c @ src.T).T + t_c
        d = np.linalg.norm(dst - transformed, axis=1)
        mask = d < inlier_threshold
        cnt = mask.sum()
        if cnt > best_count and cnt >= min_inliers:
            best_count = cnt
            best_R, best_t = estimate_rigid_transform(src[mask], dst[mask])
    return best_R, best_t

def icp_with_weights(src, tgt, w, max_iter=300, tol=1e-10):
    R, t = np.eye(2), np.zeros(2)
    prev_err = float('inf')
    for _ in range(max_iter):
        transformed = (R @ src.T).T + t
        dist, idx = NearestNeighbors(n_neighbors=1).fit(tgt).kneighbors(transformed)
        tgt_nn = tgt[idx.flatten()]
        err = (w * (dist[:,0]**2)).sum() / (w.sum() + 1e-9)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        sc = np.average(transformed, axis=0, weights=w)
        tc = np.average(tgt_nn, axis=0, weights=w)
        H = (w[:,None] * (transformed-sc)).T @ (tgt_nn - tc)
        U, _, Vt = np.linalg.svd(H)
        R_d = Vt.T @ U.T
        if np.linalg.det(R_d) < 0:
            Vt[1,:] *= -1
            R_d = Vt.T @ U.T
        t_d = tc - R_d @ sc
        R = R_d @ R
        t = R_d @ t + t_d
    return R, t, prev_err

def cpd_rigid_registration(Y, X, max_iterations=10, tolerance=1e-3):
    reg = RigidRegistration(X=X, Y=Y, max_iterations=max_iterations, tolerance=tolerance)
    Y_reg, (s, R, t) = reg.register()
    return s, R, t, Y_reg

def compute_normals(points, k=40):
    if len(points) < 3:
        return np.zeros_like(points)
    nn = min(k, len(points)-1)
    nbrs = NearestNeighbors(n_neighbors=nn).fit(points)
    _, idx = nbrs.kneighbors(points)
    normals = np.zeros_like(points)
    for i, neigh in enumerate(idx):
        cov = np.cov(points[neigh].T)
        _, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:,0]
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    return normals / norms

def normal_icp(src, tgt, w, alpha=0.1, gamma=0.8, max_iter=300, tol=1e-5):
    sn = compute_normals(src)
    tn = compute_normals(tgt)
    R, t = np.eye(2), np.zeros(2)
    prev_err = float('inf')
    for _ in range(max_iter):
        transformed = (R @ src.T).T + t
        dist, idx = NearestNeighbors(n_neighbors=1).fit(tgt).kneighbors(transformed)
        idx = idx.flatten()
        tgt_nn = tgt[idx]
        tn_nn = tn[idx]
        ndot = np.einsum('ij,ij->i', sn, tn_nn)
        n_err = np.arccos(np.clip(ndot, -1, 1))
        comb = w * dist[:,0] * np.exp(-n_err/(2*alpha))**gamma
        err = (comb * (dist[:,0]**2)).sum() / (comb.sum() + 1e-9)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        sc = np.average(transformed, axis=0, weights=comb + 1e-9)
        tc = np.average(tgt_nn, axis=0, weights=comb + 1e-9)
        H = (comb[:,None] * (transformed-sc)).T @ (tgt_nn - tc)
        U, _, Vt = np.linalg.svd(H)
        R_d = Vt.T @ U.T
        if np.linalg.det(R_d) < 0:
            Vt[1,:] *= -1
            R_d = Vt.T @ U.T
        t_d = tc - R_d @ sc
        R = R_d @ R
        t = R_d @ t + t_d
    return R, t, prev_err

def fill_line_gaps(points, max_distance=2.0, max_iter=5):
    pts = points.copy()
    for _ in range(max_iter):
        nbrs = NearestNeighbors(n_neighbors=2).fit(pts)
        dist, idx = nbrs.kneighbors(pts)
        new_pts = []
        for i in range(len(pts)):
            d = dist[i,1]
            if d > max_distance:
                p1 = pts[i]
                p2 = pts[idx[i,1]]
                n_insert = int(d // max_distance)
                step = (p2 - p1) / (n_insert + 1)
                for j in range(1, n_insert+1):
                    new_pts.append(p1 + step * j)
        if not new_pts:
            break
        pts = np.vstack([pts, np.array(new_pts)])
    return pts

def rotate_points(pts, angle, center=None):
    c = pts.mean(0) if center is None else center
    th = np.radians(angle)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (R @ (pts-c).T).T + c

def angle_from_R(R):
    ang = np.degrees(np.arctan2(R[1,0], R[0,0]))
    return ((ang+180)%360) - 180

def rotation_error(gt, est):
    return ((est-gt+180)%360) - 180
# -------------------- argparse --------------------
parser = argparse.ArgumentParser(description="Rotation Robustness Test")
parser.add_argument('--set', default='red01', help='Dataset key')
args, _ = parser.parse_known_args()
if args.set not in DATASETS:
    raise ValueError(f"未知資料集 {args.set}")
P = DATASETS[args.set]
CSV_DIR = P['csv_dir']
IMG_DIR = P['img_dir']
TEMPLATE_CSV = P['template_csv']
TEMPLATE_IMG = P['template_img']
FIRST_IMG_NAME = P['first_img']

# 將可視化圖存到指定路徑下的資料集資料夾
BASE_VIS_DIR = r"/media/dc0206/Crucial X6/GMM20/r_test"
VIS_DIR = os.path.join(BASE_VIS_DIR, args.set)
os.makedirs(VIS_DIR, exist_ok=True)


def estimate_initial_rotation(theta_src: np.ndarray, theta_ref: np.ndarray) -> float:
    """利用 GradientAngle 直方圖 FFT 對應取得最相似之循環位移。
    參數
    ------
    theta_src : np.ndarray  來源點雲角度 (deg)
    theta_ref : np.ndarray  參考點雲角度 (deg)
    回傳
    ------
    rot_init  : float 估計 src 需逆時針旋轉多少度才能與 ref 對齊, 落在 [-180, 180)
    """
    bin_edges = np.arange(0, 361)  # 1° bin
    hist_src, _ = np.histogram(theta_src % 360, bins=bin_edges)
    hist_ref, _ = np.histogram(theta_ref % 360, bins=bin_edges)
    # FFT circular correlation
    fft_src = np.fft.rfft(hist_src)
    fft_ref = np.fft.rfft(hist_ref)
    corr = np.fft.irfft(fft_src * np.conj(fft_ref))
    phi0 = int(np.argmax(corr))        # [0,359]
    rot_init = (phi0 + 180) % 360 - 180  # 轉成 [-180,180)
    return float(rot_init)

# --------------- 3. 建立或載入 ref_points & angles ---------------
REF_PT_PATH = f"ref_points_{args.set}.npy"
REF_ANG_PATH = f"ref_angles_{args.set}.npy"

if os.path.exists(REF_PT_PATH) and os.path.exists(REF_ANG_PATH):
    ref_points = np.load(REF_PT_PATH)
    ref_angles = np.load(REF_ANG_PATH)
else:
    # 由模板 CSV/影像重建
    tpl_df = pd.read_csv(TEMPLATE_CSV)
    img0 = cv2.imread(os.path.join(IMG_DIR, FIRST_IMG_NAME), cv2.IMREAD_GRAYSCALE)
    tpl_img = cv2.imread(TEMPLATE_IMG, cv2.IMREAD_GRAYSCALE)
    _, _, _, tl = cv2.minMaxLoc(cv2.matchTemplate(img0, tpl_img, cv2.TM_CCOEFF_NORMED))
    br = (tl[0] + tpl_img.shape[1], tl[1] + tpl_img.shape[0])
    df0 = pd.read_csv(os.path.join(CSV_DIR, FIRST_IMG_NAME.replace('.bmp', '.csv')))
    pts0 = df0[['PosX', 'PosY']].values.astype(np.float32)
    ang0 = df0['GradientAngle'].values.astype(np.float32)
    mask0 = (
        (pts0[:, 0] >= tl[0]) & (pts0[:, 0] <= br[0]) &
        (pts0[:, 1] >= tl[1]) & (pts0[:, 1] <= br[1])
    )
    ref0 = pts0[mask0]
    ang0 = ang0[mask0]
    ref0 = filter_noise_points_weighted(ref0)
    ang0 = ang0[:len(ref0)]  # 同步長度
    ref0 = ensure_two_points_in_circle_iterative(ref0, radius=0.5, max_iter=300)
    ang0 = ang0[:len(ref0)]
    ref0 = fill_line_gaps(ref0, max_distance=2.0)
    ang0 = np.pad(ang0, (0, len(ref0) - len(ang0)), 'edge')  # 可能新增點：用最近值填
    ref_points, ref_angles = ref0, ang0
    np.save(REF_PT_PATH, ref_points)
    np.save(REF_ANG_PATH, ref_angles)

center = ref_points.mean(0)


# ---------- 抽成工具 ----------
def get_or_build_reference(name, cfg):
    pt_path = f"ref_points_{name}.npy"
    ang_path = f"ref_angles_{name}.npy"

    if os.path.exists(pt_path) and os.path.exists(ang_path):
        return np.load(pt_path), np.load(ang_path)

    # ↓↓↓ 以下沿用你原本「else:」裡的重建流程 ↓↓↓
    tpl_df = pd.read_csv(cfg["template_csv"])
    img0 = cv2.imread(os.path.join(cfg["img_dir"], cfg["first_img"]), cv2.IMREAD_GRAYSCALE)
    tpl_img = cv2.imread(cfg["template_img"], cv2.IMREAD_GRAYSCALE)
    _, _, _, tl = cv2.minMaxLoc(cv2.matchTemplate(img0, tpl_img, cv2.TM_CCOEFF_NORMED))
    br = (tl[0] + tpl_img.shape[1], tl[1] + tpl_img.shape[0])

    df0 = pd.read_csv(os.path.join(cfg["csv_dir"], cfg["first_img"].replace('.bmp', '.csv')))
    pts0  = df0[['PosX', 'PosY']].values.astype(np.float32)
    ang0  = df0['GradientAngle'].values.astype(np.float32)

    mask0 = (
        (pts0[:, 0] >= tl[0]) & (pts0[:, 0] <= br[0]) &
        (pts0[:, 1] >= tl[1]) & (pts0[:, 1] <= br[1])
    )
    ref0   = pts0[mask0]
    ang0   = ang0[mask0]
    ref0   = filter_noise_points_weighted(ref0)
    ang0   = ang0[:len(ref0)]
    ref0   = ensure_two_points_in_circle_iterative(ref0, radius=0.5, max_iter=300)
    ang0   = ang0[:len(ref0)]
    ref0   = fill_line_gaps(ref0, max_distance=2.0)
    ang0   = np.pad(ang0, (0, len(ref0) - len(ang0)), 'edge')

    np.save(pt_path, ref0)
    np.save(ang_path, ang0)
    return ref0, ang0

# ---------- 在 process_dataset() 裡用 ----------
def process_dataset(name, cfg):
    print(f"\n=== 處理資料集: {name} ===")
    vis_dir = os.path.join(BASE_VIS_DIR, name)
    os.makedirs(vis_dir, exist_ok=True)

    ref_pts, ref_ang = get_or_build_reference(name, cfg)
    center = ref_pts.mean(0)

    records = []
    t0 = time.time()
    for gt in np.arange(0.0, 360.0, 5.0):
        # ---- 產生來源雲 (模擬真實輸入) ----
        rot_src = rotate_points(ref_pts, gt, center)            # Point
        theta_src = (ref_ang + gt) % 360                        # Angle 同步旋轉

        # ===== ORI‑HIST START =====
        rot_init = estimate_initial_rotation(theta_src, ref_ang)
        # 先反向旋轉來源雲；使其大致對齊到參考座標系
        rot = rotate_points(rot_src, -rot_init, center)         # 用於後續配準
        theta_src_adj = (theta_src - rot_init) % 360            # 調整後角度，目前僅供 debug
        # ===== ORI‑HIST END   =====

        # 密度權重
        dens = compute_local_density(rot, k=25)
        if len(dens) == 0:
            continue
        w = (dens - dens.min()) / (dens.max() - dens.min() + 1e-9)
        """
        # USAC, ICP, CPD+Normal ICP
        R_u, t_u = usac_rigid_transform(rot, ref_pts)
        R_i, t_i, e_i = normal_icp(rot, ref_pts, w + 1e-8)
        try:
            _, R_c, t_c, _ = cpd_rigid_registration(rot, ref_pts)
            Y = (R_c @ rot.T).T + t_c
        except Exception:
            R_c, t_c, Y = np.eye(2), np.zeros(2), rot.copy()
        try:
            R_n, t_n, e_n = normal_icp(Y, ref_pts, w)
        except ZeroDivisionError:
            R_n, t_n, e_n = R_i, t_i, e_i

        if e_i < e_n:
            R_f, t_f, e_f = R_i, t_i, e_i
        else:
            R_f = R_n @ R_c
            t_f = R_n @ t_c + t_n
            e_f = e_n
        """
        # 2-5 USAC
        nbrs = NearestNeighbors(n_neighbors=1).fit(ref_pts)
        _,idx2 = nbrs.kneighbors(rot)
        tgt = ref_pts[idx2.flatten()]
        try:
            R_u, t_u, _ = usac_rigid_transform(rot, tgt, inlier_threshold=40, min_inliers=20)
        except:
            R_u, t_u = np.eye(2), np.zeros(2)
        
        # 2-6 ICP + CPD + Normal-ICP
        t0 = time.time()
        R_p, t_p, e_p = normal_icp(rot, ref_pts, dens)
        try:
            _, R_c, t_c, _ = cpd_rigid_registration(rot, ref_pts, max_iterations=100, tolerance=1e-3)
        except:
            R_c, t_c = np.eye(2), np.zeros(2)
        R_n, t_n, e_n = normal_icp((R_c@rot.T).T + t_c, ref_pts, dens)
        if e_p < e_n:
            R_f, t_f, err_f = R_p, t_p, e_p
        else:
            R_f = R_n @ R_c
            t_f = R_n @ t_c + t_n
            err_f = e_n
        # ---------------- 評估 ----------------
        v_src = rot_src - center          # 真正來源向量 (未修正)
        aligned = (R_f @ rot.T).T + t_f   # 最終對齊結果 (rot 已預對齊)
        v_dst = aligned - center
        angles_pt = np.degrees(np.arctan2(v_dst[:, 1], v_dst[:, 0]) - np.arctan2(v_src[:, 1], v_src[:, 0]))
        angles_pt = ((angles_pt + 180) % 360) - 90
        from sklearn.linear_model import RANSACRegressor
        X = np.zeros((len(angles_pt), 1))
        est_angle = RANSACRegressor(residual_threshold=1.0).fit(X, angles_pt).estimator_.intercept_
        err_ang = abs(rotation_error(gt, est_angle))
        records.append([gt, est_angle, err_ang, err_f, rot_init])

        # ---- 可視化 ----
        if SAVE_VIS:
            plt.figure(figsize=(6, 6))
            plt.scatter(ref_pts[:, 0], ref_pts[:, 1], c='cyan', s=18, label='Reference')
            plt.scatter(rot_src[:, 0], rot_src[:, 1], c='red', s=8, alpha=0.4, label='Before')
            plt.scatter(aligned[:, 0], aligned[:, 1], c='lime', s=8, label='After')
            for p, q in zip(rot_src, aligned):
                plt.plot([p[0], q[0]], [p[1], q[1]], c='gray', lw=0.3)
            plt.axis('equal')
            plt.title(f"{name} GT{gt}°→Est{est_angle:.1f}° | histInit {rot_init:.1f}° | Δ{err_ang:.1f}°")
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"angle_{gt:03.0f}.png"), dpi=180)
            plt.close()

    print(f"{name} 完成 {len(records)} 次測試，耗時 {time.time() - t0:.1f}s")

    df = pd.DataFrame(records, columns=['gt', 'est', 'angle_diff', 'rmse', 'hist_init'])
    df['dataset'] = name
    print(f"{name} 完成 {len(df)} 筆測試")
    return df

# 2. 在主程式收集所有 df
if __name__ == "__main__":
    output_path = "all_rotation_test_results.xlsx"
    all_dfs = []

    for ds, cfg in DATASETS.items():
        df = process_dataset(ds, cfg)
        all_dfs.append(df)

    # 3. 合併所有 dataset 的結果
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # 4. 儲存完整結果（可選）
    df_combined.to_excel(output_path, index=False)
    print(f"已將所有結果存到 {output_path}，共 {len(df_combined)} 筆")

    # 5. **方法二**：從合併後的 df 直接抽出 rmse 欄，另存一個檔
    rmse_df = df_combined[['rmse']]              # 只保留 rmse
    rmse_df.to_excel("all_err_f.xlsx", index=False)
    print(f"已將所有 err_f（rmse）存到 all_err_f.xlsx，共 {len(rmse_df)} 筆")

