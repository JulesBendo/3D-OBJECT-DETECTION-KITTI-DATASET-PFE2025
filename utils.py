import os
import cv2
import re
import numpy as np
from typing import Tuple, Dict, List, Optional


def read_calib(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lit un fichier de calibration et extrait les matrices de transformation.

    Args:
        calib_path (str): Chemin vers le fichier de calibration texte

    Returns:
        tuple: (P2, vtc_mat) où:
            - P2: Matrice 3x4 de transformation des coordonnées caméra 3D vers pixels image 2D
            - vtc_mat: Matrice 4x4 de transformation des coordonnées LiDAR 3D vers caméra 3D
    """
    P2 = None
    vtc_mat = None
    R0 = None

    with open(calib_path) as f:
        for line in f.readlines():
            if line[:2] == "P2":
                P2 = re.split(" ", line.strip())
                P2 = np.array(P2[-12:], np.float32)
                P2 = P2.reshape((3, 4))
            if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                vtc_mat = re.split(" ", line.strip())
                vtc_mat = np.array(vtc_mat[-12:], np.float32)
                vtc_mat = vtc_mat.reshape((3, 4))
                vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
            if line[:7] == "R0_rect" or line[:6] == "R_rect":
                R0 = re.split(" ", line.strip())
                R0 = np.array(R0[-9:], np.float32)
                R0 = R0.reshape((3, 3))
                R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
                R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
    
    # Vérification que toutes les matrices nécessaires ont été trouvées
    if P2 is None or vtc_mat is None or R0 is None:
        raise ValueError(f"Impossible de lire toutes les matrices de calibration depuis {calib_path}")
        
    vtc_mat = np.matmul(R0, vtc_mat)
    return (P2, vtc_mat)


def read_velodyne(path: str, P: np.ndarray, vtc_mat: np.ndarray, IfReduce: bool = True, 
                 image_dims: Tuple[int, int] = (374, 1241)) -> np.ndarray:
    """
    Lit les données LiDAR et les filtre pour ne garder que les points visibles dans l'image.

    Args:
        path (str): Chemin vers le fichier binaire LiDAR
        P (np.array): Matrice de projection caméra 3D vers 2D
        vtc_mat (np.array): Matrice de transformation LiDAR vers caméra
        IfReduce (bool): Si True, filtre les points hors image
        image_dims (tuple): Dimensions (hauteur, largeur) de l'image cible

    Returns:
        np.array: Points LiDAR valides (coordonnées LiDAR)
    """
    max_row, max_col = image_dims  # hauteur (y), largeur (x)
    
    # Lecture du fichier binaire LiDAR
    lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))

    if not IfReduce:
        return lidar

    # Filtre uniquement les points devant la caméra (x positif)
    mask = lidar[:, 0] > 0
    lidar = lidar[mask]
    lidar_copy = np.copy(lidar)

    # Transformation du point LiDAR vers coordonnées caméra homogènes
    lidar_homogeneous = np.copy(lidar)
    lidar_homogeneous[:, 3] = 1
    
    # Application de la transformation velo_to_cam
    lidar_cam = np.matmul(lidar_homogeneous, vtc_mat.T)
    
    # Projection des points 3D sur l'image 2D
    img_pts = np.matmul(lidar_cam, P.T)
    
    # Calcul de la transformation inverse pour remettre en coordonnées LiDAR
    velo_tocam_inv = np.linalg.inv(vtc_mat)
    normal = velo_tocam_inv[0:3, 0:4]
    
    # Retour aux coordonnées LiDAR 3D
    lidar_proj = np.matmul(lidar_cam, normal.T)
    lidar_copy[:, 0:3] = lidar_proj
    
    # Calcul des coordonnées pixel
    x = img_pts[:, 0] / img_pts[:, 2]
    y = img_pts[:, 1] / img_pts[:, 2]
    
    # Création d'un masque pour les points visibles dans l'image
    mask = np.logical_and(
        np.logical_and(x >= 0, x < max_col), 
        np.logical_and(y >= 0, y < max_row)
    )

    return lidar_copy[mask]


def cam_to_velo(cloud: np.ndarray, vtc_mat: np.ndarray) -> np.ndarray:
    """
    Convertit des points 3D de coordonnées caméra vers coordonnées LiDAR.

    Args:
        cloud (np.array): Points 3D en coordonnées caméra (shape Nx3)
        vtc_mat (np.array): Matrice de transformation LiDAR vers caméra (4x4)

    Returns:
        np.array: Points 3D en coordonnées LiDAR (shape Nx3)
    """
    # Ajoute une colonne de 1 pour les coordonnées homogènes
    mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
    mat[:, 0:3] = cloud[:, 0:3]
    
    # Calcule la transformation inverse (caméra vers LiDAR)
    vtc_mat_inv = np.linalg.inv(vtc_mat)
    normal = vtc_mat_inv[0:3, 0:4]  # Ne conserve que les 3 premières lignes
    
    # Applique la transformation
    transformed_mat = np.matmul(mat, normal.T)
    return transformed_mat


def velo_to_cam(cloud: np.ndarray, vtc_mat: np.ndarray) -> np.ndarray:
    """
    Convertit des points 3D de coordonnées LiDAR vers coordonnées caméra.

    Args:
        cloud (np.array): Points 3D en coordonnées LiDAR (shape Nx3)
        vtc_mat (np.array): Matrice de transformation LiDAR vers caméra (4x4)

    Returns:
        np.array: Points 3D en coordonnées caméra (shape Nx3)
    """
    # Ajoute une colonne de 1 pour les coordonnées homogènes
    mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
    mat[:, 0:3] = cloud[:, 0:3]
    
    # Applique directement la transformation
    transformed_mat = np.matmul(mat, vtc_mat.T)
    transformed_mat = transformed_mat[:, :3]  # Supprime la composante homogène
    return transformed_mat


def read_image(path: str) -> np.ndarray:
    """
    Lit une image à partir d'un fichier.

    Args:
        path (str): Chemin vers l'image

    Returns:
        np.array: Matrice de l'image
    """
    try:
        im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        if im is None:
            # Essayer la méthode standard si imdecode échoue
            im = cv2.imread(path)
        return im
    except Exception as e:
        raise IOError(f"Impossible de lire l'image {path}: {e}")


def read_detection_label(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lit un fichier d'étiquettes de détection.

    Args:
        path (str): Chemin vers le fichier d'étiquettes

    Returns:
        tuple: (boxes, names) où:
            - boxes: Tableau des coordonnées des boîtes englobantes
            - names: Tableau des noms des étiquettes
    """
    boxes = []
    names = []

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[0]
            if this_name != "DontCare":
                line = np.array(line[-7:], np.float32)
                boxes.append(line)
                names.append(this_name)

    return np.array(boxes), np.array(names)


def read_tracking_label(path: str) -> Tuple[Dict[int, List], Dict[int, List]]:
    """
    Lit un fichier d'étiquettes de suivi (tracking).

    Args:
        path (str): Chemin vers le fichier d'étiquettes

    Returns:
        tuple: (frame_dict, names_dict) où:
            - frame_dict: Dictionnaire des objets par frame
            - names_dict: Dictionnaire des noms d'objets par frame
    """
    frame_dict = {}
    names_dict = {}

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[2]
            frame_id = int(line[0])
            ob_id = int(line[1])

            if this_name != "DontCare":
                line = np.array(line[10:17], np.float32).tolist()
                line.append(ob_id)

                if frame_id in frame_dict:
                    frame_dict[frame_id].append(line)
                    names_dict[frame_id].append(this_name)
                else:
                    frame_dict[frame_id] = [line]
                    names_dict[frame_id] = [this_name]

    return frame_dict, names_dict
