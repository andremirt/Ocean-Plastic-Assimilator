import numpy as np
import pandas as pd

from src.assimilation.cell_indexing import (
    compute_particle_ids_for_areas,
    unflatten_cell_index,
)
from src.assimilation.density_computations import (
    compute_ensemble_densities_over_parts,
)
from src.assimilation.localization import create_localization_matrix
from src.io.plotting import gen_cov_map
from src.types import AssimilatorConfig, AssimilatorDataPaths


def reintroduce_error(densities_ensemble, reinit_spreading: float, t_observation):
    for e in range(densities_ensemble.shape[0]):
        densities_ensemble[e, :, :, t_observation] += (
            np.random.randn() * reinit_spreading
        )


def flatten_density_slice(densities: np.ndarray):
    return np.moveaxis(densities, 0, -1).reshape((-1, densities.shape[0]), order="F").T


def unflatten_density_slice(
    densities_flat: np.ndarray, max_lon_id: int, max_lat_id: int
):
    return np.moveaxis(
        densities_flat.T.reshape(
            (max_lon_id, max_lat_id, densities_flat.shape[0]),
            order="F",
        ),
        -1,
        0,
    )


def compute_covariances(
    t_observation: int,
    avgs_densities: np.ndarray,
    current_densities_flat: np.ndarray,
    observations: pd.DataFrame,
    localization_matrix: np.ndarray,
    obs_cell_ids: np.ndarray,
    metrics_dir_path: str,
    config: AssimilatorConfig,
):
    n, p = avgs_densities.shape
    avgs_densities_flat = avgs_densities.reshape(-1, order="F")
    centered_densities = current_densities_flat - avgs_densities_flat[np.newaxis, :]
    cov = (
        centered_densities.T @ centered_densities[:, obs_cell_ids]
        / (current_densities_flat.shape[0] - 1)
    )
    cov *= localization_matrix
    modified_cell_ids = np.flatnonzero(np.any(localization_matrix != 0.0, axis=1))

    for observation_id, observation in enumerate(observations.itertuples()):
        lon_id = observation.lon_id
        lat_id = observation.lat_id

        if config.verbose:
            print("Computing covariances relevant for observation at", lon_id, lat_id)

        if (t_observation - config.t_start) % config.graph_plot_period == 0:
            cov_mat_for_plot = cov[:, observation_id].reshape((n, p), order="F")
            gen_cov_map(
                cov_mat_for_plot,
                "cov_mat_t"
                + str(t_observation)
                + "_"
                + str(lon_id)
                + "_"
                + str(lat_id),
                lon_id,
                lat_id,
                n,
                p,
                metrics_dir_path,
            )

    return cov, modified_cell_ids


def compute_innovation_covariance(
    cov: np.ndarray, obs_cell_ids: np.ndarray, observations: pd.DataFrame
):
    innovation_covariance = cov[obs_cell_ids, :].copy()
    observation_variances = observations["variance"].to_numpy(dtype=np.float64, copy=False)
    innovation_covariance += np.diag(observation_variances)
    return innovation_covariance


def compute_corrections(
    current_densities_flat: np.ndarray,
    cov: np.ndarray,
    observations: pd.DataFrame,
    innovation_covariance: np.ndarray,
    obs_cell_ids: np.ndarray,
    verbose: bool,
    max_lon_id: int,
    max_lat_id: int,
) -> np.ndarray:
    obs_values = observations["value"].to_numpy(dtype=np.float64, copy=False)
    if obs_values.size == 0:
        return np.zeros(
            (current_densities_flat.shape[0], max_lon_id, max_lat_id),
            dtype=np.float64,
        )

    predicted_observations = current_densities_flat[:, obs_cell_ids].T
    innovations = obs_values[:, np.newaxis] - predicted_observations

    if verbose:
        for e in range(current_densities_flat.shape[0]):
            to_correct = innovations[:, e]
            print(
                *[
                    (
                        f"Observation {obs_values[i]} and prediction {predicted_observations[i, e]} gives value to correct {to_correct[i]}"
                    )
                    for i in range(len(obs_values))
                ]
            )

    analysis_increments = np.linalg.solve(innovation_covariance, innovations)
    corrections_flat = (cov @ analysis_increments).T

    return unflatten_density_slice(
        corrections_flat,
        max_lon_id,
        max_lat_id,
    )


def update_weights(
    t_observation,
    weights,
    densities_ensemble,
    densities_ensemble_predicted,
    modified_cell_ids,
    particle_offsets,
    particle_ids,
    max_lon_id,
):
    for cell_id in modified_cell_ids:
        x, y = unflatten_cell_index(cell_id, max_lon_id)
        start = particle_offsets[cell_id]
        end = particle_offsets[cell_id + 1]

        if start == end:
            continue

        particles = particle_ids[start:end]

        totalWeight_predicted = densities_ensemble_predicted[:, x, y]
        totalWeight_corrected = densities_ensemble[:, x, y, t_observation]
        weights_predicted = np.moveaxis(weights[:, particles], 0, 1)
        scaling = np.ones_like(totalWeight_corrected)
        nonzero_predicted = totalWeight_predicted != 0
        scaling[nonzero_predicted] = (
            totalWeight_corrected[nonzero_predicted]
            / totalWeight_predicted[nonzero_predicted]
        )
        weights_corrected = weights_predicted * scaling
        weights[:, particles] = np.moveaxis(weights_corrected, 0, 1)


def assimilate(
    t_observation: int,
    densities_ensemble: np.ndarray,
    observations: pd.DataFrame,
    weights: np.ndarray,
    parts_lon: np.ndarray,
    parts_lat: np.ndarray,
    config: AssimilatorConfig,
    datapaths: AssimilatorDataPaths,
):
    # prediction at given instant
    avgs_densities = np.average(densities_ensemble[:, :, :, t_observation], axis=0)
    densities_ensemble_predicted = densities_ensemble[:, :, :, t_observation].copy()

    if config.verbose:
        print("Computing localization matrix")
    localization_matrix, obs_cell_ids = create_localization_matrix(
        config.grid_coords,
        observations,
        config.radius_observation,
    )

    if config.verbose:
        print("Computing particleIdsForAreas")
    particle_offsets, particle_ids = compute_particle_ids_for_areas(
        parts_lon, parts_lat, t_observation, config.grid_coords
    )

    if t_observation != 0 and len(observations) != 0:
        if config.verbose:
            print("Introducing model density errors")
        reintroduce_error(densities_ensemble, config.reinit_spreading, t_observation)

    current_densities_flat = flatten_density_slice(
        densities_ensemble[:, :, :, t_observation]
    )

    if config.verbose:
        print("Computing covariances")
    cov, modified_cell_ids = compute_covariances(
        t_observation,
        avgs_densities,
        current_densities_flat,
        observations,
        localization_matrix,
        obs_cell_ids,
        datapaths.metrics_dir,
        config,
    )

    if config.verbose:
        print("Computing partial Kalman Gain")
    innovation_covariance = compute_innovation_covariance(
        cov,
        obs_cell_ids,
        observations,
    )

    if config.verbose:
        print("Computing corrections")
    corrections = compute_corrections(
        current_densities_flat,
        cov,
        observations,
        innovation_covariance,
        obs_cell_ids,
        config.verbose,
        config.grid_coords.max_lon_id,
        config.grid_coords.max_lat_id,
    )
    if config.verbose:
        print("Maximum of corrections is ", corrections.max())
    if np.isnan(corrections.max()):
        return False
    densities_ensemble[:, :, :, t_observation] += corrections

    if config.verbose:
        print("Updating weights")
    update_weights(
        t_observation,
        weights,
        densities_ensemble,
        densities_ensemble_predicted,
        modified_cell_ids,
        particle_offsets,
        particle_ids,
        config.grid_coords.max_lon_id,
    )

    if config.verbose:
        print("Recomputing densities for next day")
    particle_offsets, particle_ids = compute_particle_ids_for_areas(
        parts_lon, parts_lat, t_observation + 1, config.grid_coords
    )
    compute_ensemble_densities_over_parts(
        particle_offsets,
        particle_ids,
        densities_ensemble,
        weights,
        config.grid_coords,
        config.cells_area,
        t_observation + 1,
    )

    return True
