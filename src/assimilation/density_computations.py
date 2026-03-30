from typing import List

import netCDF4 as nc
import numpy as np
from numba import njit, prange

from src.assimilation.cell_indexing import _fill_cell_ids, select_cell_id_dtype
from src.io.array_utils import to_dense_array
from src.types import RectGridCoords


def prepare_density_inputs(
    lons: np.ndarray,
    lats: np.ndarray,
    grid_coords: RectGridCoords,
    cells_area: np.ndarray,
):
    n = grid_coords.max_lon_id
    p = grid_coords.max_lat_id
    n_cells = n * p

    cell_id_dtype = select_cell_id_dtype(n_cells)
    cell_ids = np.full(
        lons.shape,
        np.iinfo(cell_id_dtype).max,
        dtype=cell_id_dtype,
        order="F",
    )
    inv_cells_area_flat = 1.0 / np.ravel(cells_area, order="F")

    inv_spacing_x = 1.0 / grid_coords.spacing_x
    inv_spacing_y = 1.0 / grid_coords.spacing_y

    _fill_cell_ids(
        cell_ids,
        lons,
        lats,
        grid_coords.x1,
        grid_coords.x2,
        grid_coords.y1,
        grid_coords.y2,
        inv_spacing_x,
        inv_spacing_y,
        n,
        p,
    )

    return cell_ids, inv_cells_area_flat, n, p, n_cells


def compute_densities(
    ds_in_path,
    ds_out_path,
    T,
    grid_coords: RectGridCoords,
    cells_area,
):
    ds_in = nc.Dataset(ds_in_path, "r")
    nb_part = ds_in["p_id"].shape[0]
    ds_out = nc.Dataset(ds_out_path, "r+")

    lons = to_dense_array(ds_in.variables["lon"][:, T], np.nan)
    lats = to_dense_array(ds_in.variables["lat"][:, T], np.nan)

    try:
        weights = to_dense_array(ds_in.variables["weight"][:], 0.0)
    except KeyError:
        weights = np.ones(nb_part, dtype=np.float64)

    cell_ids, inv_cells_area_flat, n, p, n_cells = prepare_density_inputs(
        lons,
        lats,
        grid_coords,
        cells_area,
    )

    densities_flat = np.zeros((n_cells, len(T)), dtype=np.float64, order="F")

    llvm_compute_densities(
        nb_part,
        cell_ids,
        weights,
        inv_cells_area_flat,
        densities_flat,
        n_cells,
        len(T),
    )

    densities = densities_flat.reshape((n, p, len(T)), order="F")
    ds_out["density"][:, :, T] = densities
    ds_out.close()
    ds_in.close()


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

    nb_part = parts_lon.shape[0]
    size_e = weights.shape[0]

    cell_ids, inv_cells_area_flat, n, p, n_cells = prepare_density_inputs(
        lons,
        lats,
        grid_coords,
        cells_area,
    )

    densities_flat = np.zeros(
        (n_cells, len(T), size_e),
        dtype=np.float64,
        order="F",
    )

    for e in range(size_e):
        llvm_compute_densities(
            nb_part,
            cell_ids,
            weights[e, :],
            inv_cells_area_flat,
            densities_flat[:, :, e],
            n_cells,
            len(T),
        )

    densities = densities_flat.reshape((n, p, len(T), size_e), order="F")
    all_densities[:, :, :, T] = np.moveaxis(densities, -1, 0)


def compute_ensemble_densities_over_parts(
    particle_offsets: np.ndarray,
    particle_ids: np.ndarray,
    all_densities: np.ndarray,
    weights: np.ndarray,
    grid_coords: RectGridCoords,
    cells_area: np.ndarray,
    t: int,
):
    n = grid_coords.max_lon_id
    p = grid_coords.max_lat_id
    n_cells = n * p

    inv_cells_area_flat = 1.0 / np.ravel(cells_area, order="F")
    densities_flat = np.zeros(
        (n_cells, weights.shape[0]),
        dtype=np.float64,
        order="F",
    )

    llvm_compute_ensemble_densities_over_parts(
        particle_offsets,
        particle_ids,
        weights,
        inv_cells_area_flat,
        densities_flat,
        n_cells,
        weights.shape[0],
    )

    densities = densities_flat.reshape((n, p, weights.shape[0]), order="F")
    all_densities[:, :, :, t] = np.moveaxis(densities, -1, 0)


@njit(parallel=True)
def llvm_compute_densities(
    nb_part, cell_ids, weights, inv_cells_area_flat, densities_flat, n_cells, T
):
    for t in prange(T):
        for i in range(nb_part):
            cell_id = cell_ids[i, t]

            if cell_id < n_cells:
                densities_flat[cell_id, t] += weights[i] * inv_cells_area_flat[cell_id]


@njit(parallel=True)
def llvm_compute_ensemble_densities_over_parts(
    particle_offsets,
    particle_ids,
    weights,
    inv_cells_area_flat,
    densities_flat,
    n_cells,
    size_e,
):
    for e in prange(size_e):
        for cell in range(n_cells):
            total_weight = 0.0

            for idx in range(particle_offsets[cell], particle_offsets[cell + 1]):
                total_weight += weights[e, particle_ids[idx]]

            densities_flat[cell, e] = total_weight * inv_cells_area_flat[cell]
