from typing import List
from numba.core.decorators import njit
from numba import prange
import numba.typed as nbt
import numpy as np

from src.assimilation.density_preparation import prepare_density_inputs
from src.types import RectGridCoords


def compute_ensemble_densities_over_time(
    parts_lon: np.ndarray,
    parts_lat: np.ndarray,
    all_densities: np.ndarray,
    weights: np.ndarray,
    cells_area: np.ndarray,
    T: List,
    grid_coords: RectGridCoords,
):
    lons = parts_lon[:, T]
    lats = parts_lat[:, T]

    weights = np.ma.filled(weights, 0.0) if np.ma.isMaskedArray(weights) else weights

    nbParts = parts_lon.shape[0]

    cell_ids, inv_cells_area_flat, n, p, n_cells = prepare_density_inputs(
        lons,
        lats,
        grid_coords,
        cells_area,
    )

    densities_flat = np.zeros(
        (n_cells, len(T), weights.shape[0]),
        dtype=np.float64,
        order="F",
    )

    llvm_compute_ensemble_densities_over_time(
        cell_ids,
        weights,
        inv_cells_area_flat,
        densities_flat,
        nbParts,
        n_cells,
        len(T),
        weights.shape[0],
    )

    densities = densities_flat.reshape((n, p, len(T), weights.shape[0]), order="F")
    all_densities[:, :, :, T] = np.moveaxis(densities, -1, 0)

@njit(parallel=True)
def llvm_compute_ensemble_densities_over_time(
    cell_ids,
    weights,
    inv_cells_area_flat,
    densities_flat,
    nbParts,
    n_cells,
    T,
    size_e,
):
    for e in range(size_e):
        for t in prange(T):
            for i in range(nbParts):
                cell_id = cell_ids[i, t]

                if 0 <= cell_id < n_cells:
                    densities_flat[cell_id, t, e] += (
                        weights[e, i] * inv_cells_area_flat[cell_id]
                    )


def compute_ensemble_densities_over_parts(
    partIdsForArea: nbt.List,
    all_densities: np.ndarray,
    weights: np.ndarray,
    grid_coords: RectGridCoords,
    cells_area: np.ndarray,
    t: int,
):
    densities = np.zeros(
        (grid_coords.max_lon_id, grid_coords.max_lat_id, weights.shape[0])
    )

    for x in range(grid_coords.max_lon_id):
        for y in range(grid_coords.max_lat_id):
            try:
                particleIds = partIdsForArea[x][y]
                weight = np.sum(weights[:, particleIds], axis=1)
                densities[x, y, :] += weight / cells_area[x, y]
            except KeyError:
                pass

    all_densities[:, :, :, t] = np.moveaxis(densities, -1, 0)
