import numpy as np
import netCDF4 as nc
from numba import njit, prange

from src.assimilation.density_preparation import (
    build_particle_csr,
    prepare_cell_ids_for_time,
    prepare_density_inputs,
)
from src.io.array_utils import to_dense_array
from src.types import RectGridCoords


def compute_particle_ids_for_areas(
    parts_lon: np.ndarray,
    parts_lat: np.ndarray,
    t: int,
    grid_coords: RectGridCoords,
):
    lons = parts_lon[:, t]
    lats = parts_lat[:, t]

    cell_ids, n_cells = prepare_cell_ids_for_time(
        lons,
        lats,
        grid_coords,
    )

    return build_particle_csr(cell_ids, n_cells)


def compute_densities(
    ds_in_path,
    ds_out_path,
    T,
    grid_coords: RectGridCoords,
    cells_area,
):
    ds_in = nc.Dataset(ds_in_path, "r")
    nbPart = ds_in["p_id"].shape[0]
    ds_out = nc.Dataset(ds_out_path, "r+")

    lons = to_dense_array(ds_in.variables["lon"][:, T], np.nan)
    lats = to_dense_array(ds_in.variables["lat"][:, T], np.nan)

    try:
        weights = to_dense_array(ds_in.variables["weight"][:], 0.0)
    except KeyError:
        weights = np.ones(nbPart, dtype=np.float64)

    cell_ids, inv_cells_area_flat, n, p, n_cells = prepare_density_inputs(
        lons,
        lats,
        grid_coords,
        cells_area,
    )

    densities_flat = np.zeros((n_cells, len(T)), dtype=np.float64, order="F")

    llvm_compute_densities(
        nbPart,
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

@njit(parallel=True)
def llvm_compute_densities(
    nbPart, cell_ids, weights, inv_cells_area_flat, densities_flat, n_cells, T
):
    for t in prange(T):
        for i in range(nbPart):
            cell_id = cell_ids[i, t]

            if 0 <= cell_id < n_cells:
                densities_flat[cell_id, t] += weights[i] * inv_cells_area_flat[cell_id]
