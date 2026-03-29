import numpy as np
from math import floor
import netCDF4 as nc
from numba import njit, prange
from numba.typed import List

from src.assimilation.density_preparation import prepare_density_inputs
from src.types import RectGridCoords


def compute_particle_ids_for_areas(
    parts_lon: np.ndarray,
    parts_lat: np.ndarray,
    t: int,
    grid_coords: RectGridCoords,
):
    particle_ids_for_areas = List()
    for i in range(grid_coords.max_lon_id):
        l1 = List()
        for j in range(grid_coords.max_lat_id):
            l2 = List()
            l2.append(0)
            l2.remove(0)
            l1.append(l2)
        particle_ids_for_areas.append(l1)

    lons = parts_lon[:, t]
    lats = parts_lat[:, t]

    lons = np.ma.filled(lons, 0.0) if np.ma.isMaskedArray(lons) else lons
    lats = np.ma.filled(lats, 0.0) if np.ma.isMaskedArray(lats) else lats

    return accelerated_cpfa(
        lons,
        lats,
        particle_ids_for_areas,
        x1=grid_coords.x1,
        x2=grid_coords.x2,
        y1=grid_coords.y1,
        y2=grid_coords.y2,
        spacing_x=grid_coords.spacing_x,
        spacing_y=grid_coords.spacing_y,
    )


@njit
def accelerated_cpfa(
    lons: np.ndarray,
    lats: np.ndarray,
    particle_ids_for_areas: List,
    x1: int,
    x2: int,
    y1: int,
    y2: int,
    spacing_x: float,
    spacing_y: float,
):
    for i in range(len(lons)):
        lon = lons[i]
        lat = lats[i]

        if y1 <= lat < y2 and x1 <= lon < x2:
            lonId = floor((lon - x1) / spacing_x)
            latId = floor((lat - y1) / spacing_y)

            particle_ids_for_areas[lonId][latId].append(i)

    return particle_ids_for_areas


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

    lons = ds_in.variables["lon"][:, T]
    lats = ds_in.variables["lat"][:, T]

    try:
        weights = ds_in.variables["weight"][:]
    except KeyError:
        weights = np.ones(nbPart, dtype=np.float64)

    weights = np.ma.filled(weights, 0.0) if np.ma.isMaskedArray(weights) else weights

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
