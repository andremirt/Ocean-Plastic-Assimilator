from typing import Tuple
import numpy as np
import pandas as pd

from src.assimilation.cell_indexing import flatten_cell_indices
from src.types import RectGridCoords


def create_localization_matrix(
    grid_coords: RectGridCoords, observations: pd.DataFrame, radius_observation: int
):
    n = grid_coords.max_lon_id
    p = grid_coords.max_lat_id
    n_cells = n * p

    obs_lon_ids = observations["lon_id"].to_numpy(dtype=np.int64, copy=False)
    obs_lat_ids = observations["lat_id"].to_numpy(dtype=np.int64, copy=False)
    obs_cell_ids = flatten_cell_indices(obs_lon_ids, obs_lat_ids, n)

    if len(observations) == 0:
        return np.empty((n_cells, 0), dtype=np.float64), obs_cell_ids

    x_ids, y_ids = np.meshgrid(np.arange(n), np.arange(p), indexing="ij")
    flat_x_ids = x_ids.ravel(order="F")
    flat_y_ids = y_ids.ravel(order="F")

    dx = flat_x_ids[:, np.newaxis] - obs_lon_ids[np.newaxis, :]
    dy = flat_y_ids[:, np.newaxis] - obs_lat_ids[np.newaxis, :]
    distances = np.sqrt(dx ** 2 + dy ** 2)

    if radius_observation == 0:
        localization_matrix = (distances == 0).astype(np.float64)
    else:
        localization_matrix = np.maximum(1 - distances / radius_observation, 0.0)

    return localization_matrix, obs_cell_ids


def compute_indices_circle(
    center: Tuple[int, int], radius: int, max_lon_id: int, max_lat_id: int
):
    xc, yc = center
    indices = []

    for x in range(max(xc - radius, 0), min(xc + radius + 1, max_lon_id)):
        for y in range(max(yc - radius, 0), min(yc + radius + 1, max_lat_id)):
            if np.sqrt((x - xc) ** 2 + (y - yc) ** 2) <= radius:
                indices.append((x, y))
    return indices
