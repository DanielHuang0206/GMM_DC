import os, glob, time, math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from pycpd import RigidRegistration  # pip install pycpd

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

# === 套件 ===
import os, glob, time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from pycpd import RigidRegistration  # pip install pycpd
import math
# ---------------------------
# 1. 前處理類別
# ---------------------------
class PointCloudPreprocessor:
    def __init__(self, radius=10, max_iter_circle=300, max_distance=2.0, max_iter_fill=500):
        self.radius = radius
        self.max_iter_circle = max_iter_circle
        self.max_distance = max_distance
        self.max_iter_fill = max_iter_fill

    def _ensure_two(self, pts: np.ndarray):
        new = []
        for i, c in enumerate(pts):
            d = np.linalg.norm(pts - c, axis=1)
            if np.count_nonzero(d <= self.radius) < 2:
                if len(pts) == 1 or np.allclose(d, 0):
                    rd = np.random.randn(2)
                    rd /= np.linalg.norm(rd)
                    new.append(c + rd * self.radius * 0.5)
                else:
                    si = np.argsort(d)
                    ni = si[1] if si[0] == i else si[0]
                    v = c - pts[ni]
                    nv = np.linalg.norm(v)
                    if nv < 1e-9:
                        v = np.random.randn(2); nv = np.linalg.norm(v)
                    u = v / nv
                    new.append(c + u * self.radius * 0.8)
        if new:
            return np.vstack([pts, new])
        return pts

    def _fill_gaps(self, pts: np.ndarray):
        for _ in range(self.max_iter_fill):
            nbr = NearestNeighbors(n_neighbors=2).fit(pts)
            d, idxs = nbr.kneighbors(pts)
            adds = []
            for i in range(len(pts)):
                dist = d[i,1]
                if dist > self.max_distance:
                    p1, p2 = pts[i], pts[idxs[i,1]]
                    n = int(dist // self.max_distance)
                    for k in range(1, n+1):
                        adds.append(p1 + (p2 - p1) * k/(n+1))
            if not adds:
                break
            pts = np.vstack([pts, adds])
        return pts

    def preprocess(self, pts: np.ndarray) -> np.ndarray:
        for _ in range(self.max_iter_circle):
            new = self._ensure_two(pts)
            if new.shape[0] == pts.shape[0]:
                break
            pts = new
        return self._fill_gaps(pts)

# ---------------------------
# 2. 雜訊過濾與粗對齊
# ---------------------------
class PointCloudUtils:
    @staticmethod
    def filter_noise(points: np.ndarray, eps=3, min_samples=2, min_cluster_size=55) -> np.ndarray:
        if points.size == 0:
            return points
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        lbl = db.labels_
        u, c = np.unique(lbl[lbl!=-1], return_counts=True)
        keep = u[c >= min_cluster_size]
        return points[np.isin(lbl, keep)]

    @staticmethod
    def pca_coarse_alignment(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cs, cd = src.mean(0), dst.mean(0)
        s_c, d_c = src - cs, dst - cd
        U, _, Vt = np.linalg.svd(s_c.T @ s_c)
        Ud, _, Vtd = np.linalg.svd(d_c.T @ d_c)
        R = Ud @ U.T
        if np.linalg.det(R) < 0:
            Ud[:,1] *= -1
            R = Ud @ U.T
        t = cd - R @ cs
        return R, t

# ---------------------------
# 3. 註冊：Normal-ICP 與 CPD
# ---------------------------
class PointCloudRegistration:
    def __init__(self, src: np.ndarray, tgt: np.ndarray):
        self.src = src
        self.tgt = tgt

    @staticmethod
    def compute_weights(pts: np.ndarray, k=25) -> np.ndarray:
        if pts.size == 0:
            return np.array([])
        nbr = NearestNeighbors(n_neighbors=min(k+1, len(pts))).fit(pts)
        d, _ = nbr.kneighbors(pts)
        w = 1.0 / (np.mean(d[:,1:], axis=1) + 1e-6)
        return (w - w.min()) / (w.max() - w.min() + 1e-9)

    def normal_icp(self, max_iterations=1000, tolerance=1e-9, alpha=0.1, gamma=0.8) -> Tuple[np.ndarray,np.ndarray,float]:
        R_total = np.eye(2)
        t_total = np.zeros(2)
        prev_err = float('inf')
        # 權重
        w = self.compute_weights(self.src)
        # 計算法線 for src
        nbr_s = NearestNeighbors(n_neighbors=min(40, len(self.src)-1)).fit(self.src)
        _, idxs_s = nbr_s.kneighbors(self.src)
        normals_s = np.zeros_like(self.src)
        for i, nbrs in enumerate(idxs_s):
            cov = np.cov(self.src[nbrs].T)
            _, eigvecs = np.linalg.eigh(cov)
            normals_s[i] = eigvecs[:, 0]
        # 計算法線 for tgt
        nbr_t = NearestNeighbors(n_neighbors=min(40, len(self.tgt)-1)).fit(self.tgt)
        _, idxs_t = nbr_t.kneighbors(self.tgt)
        normals_t = np.zeros_like(self.tgt)
        for i, nbrs in enumerate(idxs_t):
            cov = np.cov(self.tgt[nbrs].T)
            _, eigvecs = np.linalg.eigh(cov)
            normals_t[i] = eigvecs[:, 0]

        for _ in range(max_iterations):
            src_t = (R_total @ self.src.T).T + t_total
            nbr2 = NearestNeighbors(n_neighbors=1).fit(self.tgt)
            dists, ids = nbr2.kneighbors(src_t)
            closest = self.tgt[ids.flatten()]
            # 法線相似度
            dot = np.einsum('ij,ij->i', normals_s, normals_t[ids.flatten()])
            n_err = np.arccos(np.clip(dot, -1, 1))
            w_n = (np.exp(-n_err/(2*alpha))**gamma)
            # 結合權重
            cw = w * (dists.flatten()) * w_n
            err = np.sum(cw * (dists.flatten()**2)) / (np.sum(cw) + 1e-9)
            if abs(prev_err - err) < tolerance and err < 1:
                break
            prev_err = err
            # 加權解剛性
            src_cent = np.average(src_t, axis=0, weights=cw)
            tgt_cent = np.average(closest, axis=0, weights=cw)
            src_c = src_t - src_cent
            tgt_c = closest - tgt_cent
            H = (cw[:, None] * src_c).T @ tgt_c
            U2, _, Vt2 = np.linalg.svd(H)
            Rd = Vt2.T @ U2.T
            if np.linalg.det(Rd) < 0:
                Vt2[1, :] *= -1
                Rd = Vt2.T @ U2.T
            td = tgt_cent - Rd @ src_cent
            R_total = Rd @ R_total
            t_total = Rd @ t_total + td
        return R_total, t_total, prev_err

    def cpd_rigid(self, max_iterations=50, tolerance=1e-3):
        reg = RigidRegistration(X=self.tgt, Y=self.src,
                                max_iterations=max_iterations, tolerance=tolerance)
        Yp, (s, R, t) = reg.register()
        return s, R, t, Yp

# ---------------------------
# 4. 模板匹配與初始旋轉
# ---------------------------
class PointCloudAligner:
    def __init__(self, template: str, img_dir: str):
        self.tpl_img = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
        if self.tpl_img is None:
            raise FileNotFoundError(f"找不到模板：{template}")
        self.h, self.w = self.tpl_img.shape
        self.img_dir = img_dir

    def match(self, img_name: str) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        img = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_GRAYSCALE)
        res = cv2.matchTemplate(img, self.tpl_img, cv2.TM_CCOEFF_NORMED)
        _, _, _, tl = cv2.minMaxLoc(res)
        br = (tl[0] + self.w, tl[1] + self.h)
        return tl, br

    def get_cur_and_angles(self, idx: int, imgs: List[str], csvs: List[str], ref_ang: np.ndarray):
        img_path = imgs[idx]
        _, im = os.path.basename(img_path), cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        tl, br = self.match(os.path.basename(img_path))
        center = np.array([(tl[0]+br[0])/2, (tl[1]+br[1])/2])
        df = pd.read_csv(csvs[idx])
        pts = df[['PosX','PosY']].values
        ang = df['GradientAngle'].values
        mask = (pts[:,0]>=tl[0])&(pts[:,0]<=br[0])&(pts[:,1]>=tl[1])&(pts[:,1]<=br[1])
        cur, cur_ang = pts[mask], ang[mask]
        if len(cur)<3 or len(ref_ang)<3:
            return None, None, None, None
        return cur, cur_ang, center, os.path.basename(img_path)

    @staticmethod
    def initial_rotation(cur: np.ndarray, cur_ang: np.ndarray,
                         ref_ang: np.ndarray, center: np.ndarray,
                         thresh: float = 5.0) -> Tuple[np.ndarray, float]:
        bins = np.arange(361)
        hs, _ = np.histogram(cur_ang % 360, bins=bins)
        hr, _ = np.histogram(ref_ang % 360, bins=bins)
        corr = np.fft.irfft(np.fft.rfft(hs) * np.conj(np.fft.rfft(hr)))
        phi = (int(np.argmax(corr)) + 180) % 360 - 180
        theta = math.radians(phi)
        if abs(theta) > math.radians(thresh):
            R0 = np.array([[math.cos(theta), -math.sin(theta)],
                           [math.sin(theta),  math.cos(theta)]])
            cur = (R0 @ (cur - center).T).T + center
        return cur, theta

    @staticmethod
    def append(results: List[Dict], image_file: str,
               R: np.ndarray, t: np.ndarray,
               inc: np.ndarray, disp: np.ndarray):
        results.append({
            'image_file': image_file,
            'rotation_matrix': R.tolist(),
            'translation_vector': t.tolist(),
            'incremental_displacement': inc.tolist(),
            'incremental_disp_norm': float(np.linalg.norm(inc)),
            'displacement': disp.tolist(),
            'displacement_norm': float(np.linalg.norm(disp)),
        })

# ---------------------------
# 5. 主流程與輔助函式
# ---------------------------
def get_or_build_reference(name: str, cfg: Dict,
                           do_template: bool = True,
                           do_gap_fill: bool = True) -> Tuple[np.ndarray,np.ndarray]:
    ptf = f"ref_pts_{name}.npy"
    angf = f"ref_ang_{name}.npy"
    if os.path.exists(ptf) and os.path.exists(angf):
        return np.load(ptf), np.load(angf)
    aligner = PointCloudAligner(cfg['template_img'], cfg['img_dir'])
    img0 = cv2.imread(os.path.join(cfg['img_dir'], cfg['first_img']), cv2.IMREAD_GRAYSCALE)
    tl, br = aligner.match(cfg['first_img']) if do_template else ((0,0),(img0.shape[1],img0.shape[0]))
    df0 = pd.read_csv(os.path.join(cfg['csv_dir'], cfg['first_img'].replace('.bmp','.csv')))
    pts0 = df0[['PosX','PosY']].values
    ang0 = df0['GradientAngle'].values
    mask0 = (pts0[:,0]>=tl[0])&(pts0[:,0]<=br[0])&(pts0[:,1]>=tl[1])&(pts0[:,1]<=br[1])
    ref_pts, ref_ang = pts0[mask0], ang0[mask0]
    ref_pts = PointCloudUtils.filter_noise(ref_pts)
    if do_gap_fill:
        preproc = PointCloudPreprocessor()
        ref_pts = preproc.preprocess(ref_pts)
        if len(ref_ang)<len(ref_pts): ref_ang = np.pad(ref_ang,(0,len(ref_pts)-len(ref_ang)),'edge')
    np.save(ptf,ref_pts); np.save(angf,ref_ang)
    return ref_pts, ref_ang


def process_all_images(cfg: Dict, imgs: List[str], csvs: List[str],
                       ref: np.ndarray, ref_ang: np.ndarray,
                       do_template=True, do_preprocess=True,
                       do_rotation=True, do_coarse=True,
                       do_fine_icp=True, do_fine_cpd=True) -> Tuple[List[Dict],int,int]:
    """
    處理所有影像，可選擇執行 ICP、CPD 或兩者比較後選擇誤差較小者。
    返回: results, icp_count, cpd_count
    """
    results, icp_c, cpd_c = [], 0, 0
    aligner = PointCloudAligner(cfg['template_img'], cfg['img_dir'])
    preproc = PointCloudPreprocessor()
    prev_inc = np.zeros(2)
    for i in range(1, len(imgs)):
        fn = os.path.basename(imgs[i])
        im = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        # ROI 與點雲讀取
        if do_template:
            cur, cur_ang, center, fn2 = aligner.get_cur_and_angles(i, imgs, csvs, ref_ang)
            if cur is None:
                continue
            fn = fn2
        else:
            center = np.array([im.shape[1]/2, im.shape[0]/2])
            df = pd.read_csv(csvs[i])
            cur = df[['PosX', 'PosY']].values
            cur_ang = df['GradientAngle'].values
            if len(cur) < 3:
                continue
        # 前處理
        if do_preprocess:
            cur = PointCloudUtils.filter_noise(cur)
            cur = preproc.preprocess(cur)
        # 初始旋轉
        if do_rotation:
            cur, _ = aligner.initial_rotation(cur, cur_ang, ref_ang, center)
        # 粗對齊
        R_co, t_co = PointCloudUtils.pca_coarse_alignment(cur, ref)
        cur_co = (R_co @ cur.T).T + t_co
        # 精細對齊
        R_final = np.eye(2)
        t_final = np.zeros(2)
        chosen = 'none'
        # 同時選擇 ICP & CPD
        if do_fine_icp and do_fine_cpd:
            # 執行 Normal-ICP
            reg_icp = PointCloudRegistration(cur_co, ref)
            R_icp, t_icp, err_icp = reg_icp.normal_icp()
            # 執行 CPD + Normal-ICP
            reg_cpd = PointCloudRegistration(cur_co, ref)
            try:
                _, R_cpd, t_cpd, _ = reg_cpd.cpd_rigid(max_iterations=10, tolerance=1e-3)
            except:
                R_cpd, t_cpd = np.eye(2), np.zeros(2)
            cur_cpd = (R_cpd @ cur_co.T).T + t_cpd
            reg2 = PointCloudRegistration(cur_cpd, ref)
            R_n, t_n, err_n = reg2.normal_icp()
            # 比較誤差
            if err_icp <= err_n:
                R_final, t_final = R_icp, t_icp
                icp_c += 1
                chosen = 'ICP'
            else:
                R_final = R_n @ R_cpd
                t_final = R_n @ t_cpd + t_n
                cpd_c += 1
                chosen = 'CPD'
        elif do_fine_icp:
            reg_icp = PointCloudRegistration(cur_co, ref)
            R_final, t_final, _ = reg_icp.normal_icp()
            icp_c += 1
            chosen = 'ICP'
        elif do_fine_cpd:
            reg_cpd = PointCloudRegistration(cur_co, ref)
            try:
                _, R_cpd, t_cpd, _ = reg_cpd.cpd_rigid(max_iterations=10, tolerance=1e-1)
            except:
                R_cpd, t_cpd = np.eye(2), np.zeros(2)
            cur_cpd = (R_cpd @ cur_co.T).T + t_cpd
            reg_icp2 = PointCloudRegistration(cur_cpd, ref)
            R_n, t_n, _ = reg_icp2.normal_icp()
            R_final = R_n @ R_cpd
            t_final = R_n @ t_cpd + t_n
            cpd_c += 1
            chosen = 'CPD'
        # 合併變換
        R_tot = R_final @ R_co
        t_tot = R_final @ t_co + t_final
        # 更新點與位移
        tc = R_tot @ center + t_tot
        inc = tc - center
        disp = inc - prev_inc
        prev_inc = inc.copy()
        # 收集結果，包括選擇方法
        aligner.append(results, fn, R_tot, t_tot, inc, disp)
    return results, icp_c, cpd_c


def process_dataset(cfg: Dict):
    ref_pts,ref_ang = get_or_build_reference(cfg['name'],cfg,do_template=True,do_gap_fill=True)
    imgs = sorted(glob.glob(os.path.join(cfg['img_dir'],'*.bmp')))
    csvs = sorted(glob.glob(os.path.join(cfg['csv_dir'],'*.csv')))
    results=[]
    PointCloudAligner.append(results,cfg['first_img'],np.eye(2),np.zeros(2),np.zeros(2),np.zeros(2))
    more,icp_n,cpd_n = process_all_images(cfg,imgs,csvs,ref_pts,ref_ang,do_template=True)
    results.extend(more)
    print(f"ICP 被選 {icp_n} 次，CPD 被選 {cpd_n} 次")
    os.makedirs(os.path.dirname(cfg['final_xlsx']),exist_ok=True)
    pd.DataFrame(results).to_excel(cfg['final_xlsx'],index=False)

# === 主流程 ===
if __name__=='__main__':
    for cfg in DATASETS:
        print(f"Processing [{cfg['name']}]...")
        try:
            st=time.time()
            process_dataset(cfg)
            print(f"✅ [{cfg['name']}] 完成，耗時 {time.time()-st:.1f}s")
        except Exception as e:
            print(f"[{cfg['name']}] 失敗：{e}")