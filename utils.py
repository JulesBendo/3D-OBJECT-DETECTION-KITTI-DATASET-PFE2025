import os
import cv2
import re
import numpy as np
from typing import Tuple, List, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt


def load_calibration(calib_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Fichier de calibration {calib_path} introuvable.")

    P2 = None
    vtc_mat = None
    R0 = None

    with calib_path.open('r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            key, *values = re.split(r'\s+', line)
            values = np.array(values, dtype=np.float32)

            if key == "P2":
                P2 = values.reshape(3, 4)
            elif key in ["Tr_velo_to_cam", "Tr_velo_cam"]:
                vtc_mat = np.eye(4, dtype=np.float32)
                vtc_mat[:3, :4] = values.reshape(3, 4)
            elif key in ["R0_rect", "R_rect"]:
                R0 = np.eye(4, dtype=np.float32)
                R0[:3, :3] = values.reshape(3, 3)

    if P2 is None or vtc_mat is None or R0 is None:
        missing = []
        if P2 is None: missing.append("P2")
        if vtc_mat is None: missing.append("Tr_velo_to_cam")
        if R0 is None: missing.append("R0_rect")
        raise ValueError(f"Matrices manquantes dans {calib_path}: {', '.join(missing)}")

    return P2, R0 @ vtc_mat


def load_velodyne_points(path: Union[str, Path],
                         P: np.ndarray,
                         vtc_mat: np.ndarray,
                         filter_image: bool = True,
                         image_size: Tuple[int, int] = (374, 1241),
                         min_dist: float = 0.1) -> np.ndarray:
    path = Path(path)
    try:
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

        if not filter_image:
            return points

        points = points[points[:, 0] > min_dist]
        if len(points) == 0:
            return np.empty((0, 4), dtype=np.float32)

        hom_points = np.column_stack((points[:, :3], np.ones(points.shape[0])))
        cam_points = hom_points @ vtc_mat.T

        proj = cam_points @ P.T
        proj[:, :2] /= proj[:, 2:3]

        h, w = image_size
        mask = (
            (proj[:, 0] >= 0) & (proj[:, 0] < w) &
            (proj[:, 1] >= 0) & (proj[:, 1] < h) &
            (proj[:, 2] > 0)
        )

        return points[mask]

    except Exception as e:
        raise IOError(f"Erreur lors de la lecture du fichier LiDAR {path}: {e}")


def load_image(image_path: Union[str, Path], color_mode: str = 'rgb') -> np.ndarray:
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de lire l'image {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if color_mode.lower() == 'rgb' else img


def load_labels(label_path: Union[str, Path], keep_classes: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    label_path = Path(label_path)
    boxes, names = [], []

    try:
        with label_path.open('r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] == "DontCare":
                    continue

                class_name = parts[0]
                if keep_classes is None or class_name in keep_classes:
                    boxes.append(np.array(parts[-7:], dtype=np.float32))
                    names.append(class_name)

        return (np.array(boxes) if boxes else np.empty((0, 7), dtype=np.float32),
                np.array(names, dtype=object) if names else np.empty(0, dtype=object))
    except Exception as e:
        raise IOError(f"Erreur lors de la lecture des labels {label_path}: {e}")


def transform_points(points: np.ndarray, transform: np.ndarray, inverse: bool = False) -> np.ndarray:
    if points.size == 0:
        return points.copy()

    if points.shape[1] == 3:
        points = np.column_stack((points, np.ones(points.shape[0])))

    if inverse:
        transform = np.linalg.inv(transform)

    return (points @ transform.T)[:, :3]


# Aliases
velo_to_cam = lambda pts, mat: transform_points(pts, mat, inverse=False)
cam_to_velo = lambda pts, mat: transform_points(pts, mat, inverse=True)


def overlay_lidar_on_image(points: np.ndarray,
                           image: np.ndarray,
                           P: np.ndarray,
                           vtc_mat: np.ndarray,
                           size: int = 2,
                           colormap: str = 'jet') -> np.ndarray:
    if points.size == 0:
        return image.copy()

    hom_points = np.column_stack((points[:, :3], np.ones(points.shape[0])))
    cam_points = hom_points @ vtc_mat.T
    img_coords = cam_points @ P.T
    img_coords[:, :2] /= img_coords[:, 2:3]

    # Use distance or intensity
    if colormap == 'intensity' and points.shape[1] >= 4:
        values = points[:, 3]
    else:
        values = np.linalg.norm(points[:, :3], axis=1)

    values = (values - values.min()) / (values.ptp() + 1e-6)
    colors = plt.get_cmap(colormap)(values)[:, :3] * 255

    result = image.copy()
    for (x, y), color in zip(img_coords[:, :2].astype(int), colors.astype(np.uint8)):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(result, (x, y), size, color.tolist(), -1)

    return result
