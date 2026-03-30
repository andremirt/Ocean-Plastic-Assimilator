import numpy as np
from numba import njit, prange

from src.types import RectGridCoords


@njit(inline="always")
def flatten_cell_index(x: int, y: int, max_lon_id: int):
    return x + max_lon_id * y


def flatten_cell_indices(x_ids, y_ids, max_lon_id: int):
    return np.asarray(x_ids) + max_lon_id * np.asarray(y_ids)


def unflatten_cell_index(cell_id: int, max_lon_id: int):
    return cell_id % max_lon_id, cell_id // max_lon_id


def select_cell_id_dtype(n_cells: int):
    if n_cells <= np.iinfo(np.uint8).max:
        return np.uint8
    if n_cells <= np.iinfo(np.uint16).max:
        return np.uint16
    if n_cells <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def prepare_cell_ids_for_time(
    lons: np.ndarray,
    lats: np.ndarray,
    grid_coords: RectGridCoords,
):
    n = grid_coords.max_lon_id
    p = grid_coords.max_lat_id
    n_cells = n * p

    cell_id_dtype = select_cell_id_dtype(n_cells)
    cell_ids = np.full(lons.shape, np.iinfo(cell_id_dtype).max, dtype=cell_id_dtype)

    inv_spacing_x = 1.0 / grid_coords.spacing_x
    inv_spacing_y = 1.0 / grid_coords.spacing_y

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
        n,
        p,
    )

    return cell_ids, n_cells


def build_particle_csr(cell_ids: np.ndarray, n_cells: int):
    return _build_particle_csr(cell_ids, n_cells)


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
    n,
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
                cell_ids[i, t] = flatten_cell_index(lon_id, lat_id, n)


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
    n,
    p,
):
    nb_part = lons.shape[0]

    for i in range(nb_part):
        lon = lons[i]
        lat = lats[i]

        if x1 <= lon < x2 and y1 <= lat < y2:
            lon_id = int((lon - x1) * inv_spacing_x)
            lat_id = int((lat - y1) * inv_spacing_y)
            cell_ids[i] = flatten_cell_index(lon_id, lat_id, n)


@njit
def _build_particle_csr(cell_ids, n_cells):
    counts = np.zeros(n_cells, dtype=np.int32)

    for i in range(cell_ids.shape[0]):
        cell_id = cell_ids[i]

        if cell_id < n_cells:
            counts[cell_id] += 1

    offsets = np.empty(n_cells + 1, dtype=np.int32)
    offsets[0] = 0

    for cell in range(n_cells):
        offsets[cell + 1] = offsets[cell] + counts[cell]

    particle_ids = np.empty(offsets[-1], dtype=np.int32)
    write_offsets = offsets[:-1].copy()

    for i in range(cell_ids.shape[0]):
        cell_id = cell_ids[i]

        if cell_id < n_cells:
            write_index = write_offsets[cell_id]
            particle_ids[write_index] = i
            write_offsets[cell_id] += 1

    return offsets, particle_ids
