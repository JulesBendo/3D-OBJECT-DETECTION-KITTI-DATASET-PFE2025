import os
import cv2
import re
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path

def read_calib(calib_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lit un fichier de calibration et extrait les matrices de transformation avec une meilleure gestion des erreurs.
    
    Args:
        calib_path: Chemin vers le fichier de calibration texte
        
    Returns:
        tuple: (P2, vtc_mat) où:
            - P2: Matrice 3x4 de projection caméra
            - vtc_mat: Matrice 4x4 de transformation LiDAR vers caméra
            
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si les matrices ne peuvent pas être lues
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Fichier de calibration {calib_path} introuvable")
        
    P2 = None
    vtc_mat = None
    R0 = None
    
    try:
        with open(calib_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                key, *values = re.split(r'\s+', line)
                values = np.array(values, dtype=np.float32)
                
                if key == "P2":
                    P2 = values[-12:].reshape(3, 4)
                elif key in ["Tr_velo_to_cam", "Tr_velo_cam"]:
                    vtc_mat = np.eye(4)
                    vtc_mat[:3, :4] = values[-12:].reshape(3, 4)
                elif key in ["R0_rect", "R_rect"]:
                    R0 = np.eye(4)
                    R0[:3, :3] = values[-9:].reshape(3, 3)
    
        if P2 is None or vtc_mat is None or R0 is None:
            missing = []
            if P2 is None: missing.append("P2")
            if vtc_mat is None: missing.append("Tr_velo_to_cam")
            if R0 is None: missing.append("R0_rect")
            raise ValueError(f"Matrices manquantes dans {calib_path}: {', '.join(missing)}")
            
        vtc_mat = R0 @ vtc_mat  # Multiplication matricielle plus lisible avec @
        return P2, vtc_mat
        
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture de {calib_path}: {str(e)}")


def read_velodyne(path: Union[str, Path], 
                 P: np.ndarray, 
                 vtc_mat: np.ndarray, 
                 IfReduce: bool = True, 
                 image_dims: Tuple[int, int] = (374, 1241),
                 min_distance: float = 0.1) -> np.ndarray:
    """
    Version optimisée de la lecture LiDAR avec filtres supplémentaires.
    
    Args:
        path: Chemin vers le fichier binaire LiDAR
        P: Matrice de projection 3D->2D
        vtc_mat: Matrice de transformation LiDAR->caméra
        IfReduce: Filtre les points hors image si True
        image_dims: Dimensions (h, w) de l'image
        min_distance: Distance minimale pour filtrer les points proches
        
    Returns:
        Points LiDAR filtrés (Nx4)
    """
    try:
        # Lecture plus robuste des données LiDAR
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        
        if not IfReduce:
            return points
            
        # Filtrage des points derrière la caméra ou trop proches
        front_mask = points[:, 0] > min_distance
        points = points[front_mask]
        
        if points.size == 0:
            return np.empty((0, 4), dtype=np.float32)
            
        # Transformation vers coordonnées caméra
        hom_points = np.column_stack([points[:, :3], np.ones(points.shape[0])])
        cam_points = hom_points @ vtc_mat.T
        
        # Projection sur l'image
        img_points = cam_points @ P.T
        img_points[:, :2] /= img_points[:, 2:3]  # Normalisation
        
        # Masque des points visibles
        h, w = image_dims
        in_image_mask = ((img_points[:, 0] >= 0) & (img_points[:, 0] < w) & \
                       ((img_points[:, 1] >= 0) & (img_points[:, 1] < h) & \
                       (img_points[:, 2] > 0)  # Devant la caméra
                       
        return points[in_image_mask]
        
    except Exception as e:
        raise IOError(f"Erreur lors du traitement de {path}: {str(e)}")


def read_image(path: Union[str, Path], 
              mode: str = 'rgb') -> np.ndarray:
    """
    Lecture d'image avec options supplémentaires.
    
    Args:
        path: Chemin vers l'image
        mode: 'rgb' ou 'bgr' pour l'ordre des canaux
        
    Returns:
        Image en format numpy array
    """
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Impossible de lire l'image {path}")
            
        if mode.lower() == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        raise IOError(f"Erreur lors de la lecture de {path}: {str(e)}")


def read_detection_label(path: Union[str, Path], 
                        filter_classes: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lecture des labels avec filtrage des classes.
    
    Args:
        path: Chemin vers le fichier label
        filter_classes: Liste des classes à conserver (None pour toutes)
        
    Returns:
        (boxes, names) où boxes sont les annotations 3D
    """
    boxes = []
    names = []
    
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                cls_name = parts[0]
                if cls_name == "DontCare":
                    continue
                    
                if filter_classes is None or cls_name in filter_classes:
                    boxes.append(np.array(parts[-7:], dtype=np.float32))
                    names.append(cls_name)
                    
        return np.array(boxes) if boxes else np.empty((0, 7), dtype=np.float32), \
               np.array(names) if names else np.empty(0, dtype=object)
               
    except Exception as e:
        raise IOError(f"Erreur lors de la lecture de {path}: {str(e)}")


# Fonctions de conversion optimisées
def transform_points(points: np.ndarray, 
                    transformation: np.ndarray,
                    direction: str = 'forward') -> np.ndarray:
    """
    Transformation générique des points 3D.
    
    Args:
        points: Points 3D (Nx3 ou Nx4)
        transformation: Matrice de transformation 4x4
        direction: 'forward' ou 'inverse'
        
    Returns:
        Points transformés
    """
    if points.size == 0:
        return points.copy()
        
    if points.shape[1] == 3:
        points = np.column_stack([points, np.ones(points.shape[0])])
        
    if direction == 'inverse':
        transformation = np.linalg.inv(transformation)
        
    return (points @ transformation.T)[:, :3]


# Alias pour compatibilité
cam_to_velo = lambda cloud, vtc_mat: transform_points(cloud, vtc_mat, 'inverse')
velo_to_cam = lambda cloud, vtc_mat: transform_points(cloud, vtc_mat, 'forward')


def visualize_lidar_on_image(points: np.ndarray, 
                           image: np.ndarray,
                           P: np.ndarray,
                           vtc_mat: np.ndarray,
                           point_size: int = 2,
                           color_map: str = 'jet') -> np.ndarray:
    """
    Visualisation des points LiDAR projetés sur l'image.
    """
    if points.size == 0:
        return image.copy()
        
    # Transformation des points
    hom_points = np.column_stack([points[:, :3], np.ones(points.shape[0])])
    cam_points = hom_points @ vtc_mat.T
    img_points = cam_points @ P.T
    img_points[:, :2] /= img_points[:, 2:3]
    
    # Création d'une colormap basée sur l'intensité ou la distance
    if color_map == 'intensity' and points.shape[1] >= 4:
        colors = points[:, 3]
    else:
        distances = np.linalg.norm(points[:, :3], axis=1)
        colors = distances
        
    # Normalisation des couleurs
    colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
    colors = plt.get_cmap(color_map)(colors)[:, :3] * 255
    
    # Dessin des points
    img_viz = image.copy()
    for (u, v), color in zip(img_points[:, :2].astype(int), colors):
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            cv2.circle(img_viz, (u, v), point_size, color.tolist(), -1)
            
    return img_viz
