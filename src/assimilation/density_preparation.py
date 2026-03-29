import numpy as np
from numba import njit, prange

from src.types import RectGridCoords


def select_cell_id_dtype(n_cells: int):
    if n_cells <= np.iinfo(np.int8).max + 1:
        return np.int8
    if n_cells <= np.iinfo(np.int16).max + 1:
        return np.int16
    if n_cells <= np.iinfo(np.int32).max + 1:
        return np.int32
    return np.int64


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

    cell_ids = np.full(lons.shape, -1, dtype=cell_id_dtype, order="F")
    inv_cells_area_flat = 1.0 / np.ravel(cells_area, order="C")

    inv_spacing_x = 1.0 / grid_coords.spacing_x
    inv_spacing_y = 1.0 / grid_coords.spacing_y

    if np.ma.isMaskedArray(lons) or np.ma.isMaskedArray(lats):
        lon_data = np.ma.getdata(lons)
        lat_data = np.ma.getdata(lats)
        lon_mask = np.ma.getmaskarray(lons)
        lat_mask = np.ma.getmaskarray(lats)

        _fill_cell_ids_with_masks(
            cell_ids,
            lon_data,
            lat_data,
            lon_mask,
            lat_mask,
            grid_coords.x1,
            grid_coords.x2,
            grid_coords.y1,
            grid_coords.y2,
            inv_spacing_x,
            inv_spacing_y,
            p,
        )
    else:
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
            p,
        )

    return cell_ids, inv_cells_area_flat, n, p, n_cells


def prepare_cell_ids_for_time(
    lons: np.ndarray,
    lats: np.ndarray,
    grid_coords: RectGridCoords,
):
    n = grid_coords.max_lon_id
    p = grid_coords.max_lat_id
    n_cells = n * p

    cell_id_dtype = select_cell_id_dtype(n_cells)
    cell_ids = np.full(lons.shape, -1, dtype=cell_id_dtype)

    inv_spacing_x = 1.0 / grid_coords.spacing_x
    inv_spacing_y = 1.0 / grid_coords.spacing_y

    if np.ma.isMaskedArray(lons) or np.ma.isMaskedArray(lats):
        lon_data = np.ma.getdata(lons)
        lat_data = np.ma.getdata(lats)
        lon_mask = np.ma.getmaskarray(lons)
        lat_mask = np.ma.getmaskarray(lats)

        _fill_cell_ids_for_time_with_masks(
            cell_ids,
            lon_data,
            lat_data,
            lon_mask,
            lat_mask,
            grid_coords.x1,
            grid_coords.x2,
            grid_coords.y1,
            grid_coords.y2,
            inv_spacing_x,
            inv_spacing_y,
            p,
        )
    else:
        _fill_cell_ids_for_time(
            cell_ids,
            lons,
            lats,
            grid_coords.x1,
            grid_coords.x2,
            grid_coords.y1,
            grid_coords.y2,
            inv_spacing_x,
            inv_spacing_y,
            p,
        )

    return cell_ids, n_cells


def build_particle_csr(cell_ids: np.ndarray, n_cells: int):
    return _build_particle_csr(cell_ids, n_cells)


@njit(parallel=True)
def _fill_cell_ids(
    cell_ids,
    lons,
    lats,
    x1,
    x2,
    y1,
    y2,
    inv_spacing_x,
    inv_spacing_y,
    p,
):
    nb_part, T = lons.shape

    for t in prange(T):
        for i in range(nb_part):
            lon = lons[i, t]
            lat = lats[i, t]

            if x1 <= lon < x2 and y1 <= lat < y2:
                lon_id = int((lon - x1) * inv_spacing_x)
                lat_id = int((lat - y1) * inv_spacing_y)
                cell_ids[i, t] = lon_id * p + lat_id


@njit(parallel=True)
def _fill_cell_ids_with_masks(
    cell_ids,
    lons,
    lats,
    lon_mask,
    lat_mask,
    x1,
    x2,
    y1,
    y2,
    inv_spacing_x,
    inv_spacing_y,
    p,
):
    nb_part, T = lons.shape

    for t in prange(T):
        for i in range(nb_part):
            if lon_mask[i, t] or lat_mask[i, t]:
                continue

            lon = lons[i, t]
            lat = lats[i, t]

            if x1 <= lon < x2 and y1 <= lat < y2:
                lon_id = int((lon - x1) * inv_spacing_x)
                lat_id = int((lat - y1) * inv_spacing_y)
                cell_ids[i, t] = lon_id * p + lat_id


@njit
def _fill_cell_ids_for_time(
    cell_ids,
    lons,
    lats,
    x1,
    x2,
    y1,
    y2,
    inv_spacing_x,
    inv_spacing_y,
    p,
):
    nb_part = lons.shape[0]

    for i in range(nb_part):
        lon = lons[i]
        lat = lats[i]

        if x1 <= lon < x2 and y1 <= lat < y2:
            lon_id = int((lon - x1) * inv_spacing_x)
            lat_id = int((lat - y1) * inv_spacing_y)
            cell_ids[i] = lon_id * p + lat_id


@njit
def _fill_cell_ids_for_time_with_masks(
    cell_ids,
    lons,
    lats,
    lon_mask,
    lat_mask,
    x1,
    x2,
    y1,
    y2,
    inv_spacing_x,
    inv_spacing_y,
    p,
):
    nb_part = lons.shape[0]

    for i in range(nb_part):
        if lon_mask[i] or lat_mask[i]:
            continue

        lon = lons[i]
        lat = lats[i]

        if x1 <= lon < x2 and y1 <= lat < y2:
            lon_id = int((lon - x1) * inv_spacing_x)
            lat_id = int((lat - y1) * inv_spacing_y)
            cell_ids[i] = lon_id * p + lat_id


@njit
def _build_particle_csr(cell_ids, n_cells):
    counts = np.zeros(n_cells, dtype=np.int32)

    for i in range(cell_ids.shape[0]):
        cell_id = cell_ids[i]

        if 0 <= cell_id < n_cells:
            counts[cell_id] += 1

    offsets = np.empty(n_cells + 1, dtype=np.int32)
    offsets[0] = 0

    for cell in range(n_cells):
        offsets[cell + 1] = offsets[cell] + counts[cell]

    particle_ids = np.empty(offsets[-1], dtype=np.int32)
    write_offsets = offsets[:-1].copy()

    for i in range(cell_ids.shape[0]):
        cell_id = cell_ids[i]

        if 0 <= cell_id < n_cells:
            write_index = write_offsets[cell_id]
            particle_ids[write_index] = i
            write_offsets[cell_id] += 1

    return offsets, particle_ids
