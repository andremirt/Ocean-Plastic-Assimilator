import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from src.io.Metrics import Metrics
from src.io.array_utils import to_dense_array
from src.assimilation.sampling import sample_observations
from src.assimilation.assimilate import assimilate

from src.types import AssimilatorConfig, AssimilatorDataPaths, ObservationsType


def start_simulation(datapaths: AssimilatorDataPaths, config: AssimilatorConfig):
    if config.verbose:
        print("Opening necessary datasets")
    ds_parts_ensembles = nc.Dataset(datapaths.ds_parts_ensemble, "r")
    weights = to_dense_array(ds_parts_ensembles["weight"][:, :], 0.0)
    parts_lon = to_dense_array(ds_parts_ensembles["lon"][:, :], np.nan)
    parts_lat = to_dense_array(ds_parts_ensembles["lat"][:, :], np.nan)
    ds_parts_ensembles.close()

    ds_densities_ensemble = nc.Dataset(datapaths.ds_densities_ensemble, "r")
    densities_ensemble = to_dense_array(ds_densities_ensemble["density"][:, :, :, :], 0.0)
    ds_densities_ensemble.close()

    parts_ref_lon = None
    parts_ref_lat = None
    weights_ref = None

    if config.observations.type == ObservationsType.from_simulation:
        if config.verbose:
            print("Getting reference data")
        ds_densities_ref = nc.Dataset(datapaths.ds_densities_ref)
        densities_ref = to_dense_array(ds_densities_ref["density"][:, :, :], 0.0)
        ds_densities_ref.close()

        ds_parts_ref = nc.Dataset(config.observations.ds_reference_path, "r")
        parts_ref_lon = to_dense_array(ds_parts_ref["lon"][:, :], np.nan)
        parts_ref_lat = to_dense_array(ds_parts_ref["lat"][:, :], np.nan)

        try:
            weights_ref = to_dense_array(ds_parts_ref["weight"][:], 0.0)
        except KeyError:
            weights_ref = np.ones(parts_ref_lon.shape[0], dtype=np.float64)

        ds_parts_ref.close()
    else:
        densities_ref = None

    # =================================================== INITIAL METRICS ========================================================
    if config.verbose:
        print("Generating initial metrics")

    metrics = Metrics(
        datapaths.metrics_dir,
        densities_ensemble,
        config.max_time,
        config.observations.type,
        config.grid_coords,
        parts_ref_lon=parts_ref_lon,
        parts_ref_lat=parts_ref_lat,
        weights_ref=weights_ref,
    )
    metrics.log_metrics(
        densities_ensemble,
        densities_ref,
        weights,
        parts_lon,
        parts_lat,
        config.t_start,
    )

    # =================================================== ITERATIONS ========================================================

    try:
        print("Start iterating")
        for t in range(config.t_start, config.t_end):
            print("=================================================")
            print(
                f"Start iteration {t - config.t_start + 1} / {config.t_end - config.t_start}"
            )

            if config.observations.type == ObservationsType.from_simulation:
                if config.verbose:
                    print("Sampling observations from reference simulation")
                observations = sample_observations(
                    densities_ref,
                    config.observations.locations,
                    config.observations.error_percent,
                    config.observations.measure_resolution,
                    t,
                )
            else:
                if config.verbose:
                    print("Retrieving observations from csv")
                df_observations = config.observations.df
                observations: pd.DataFrame = df_observations[
                    df_observations["time"] == t
                ]

            assimilate_res = assimilate(
                t,
                densities_ensemble,
                observations,
                weights,
                parts_lon,
                parts_lat,
                config,
                datapaths,
            )

            if not assimilate_res:
                break

            metrics.log_metrics(
                densities_ensemble,
                densities_ref,
                weights,
                parts_lon,
                parts_lat,
                t + 1,
            )
            if (t - config.t_start) % config.graph_plot_period == 0:
                if config.verbose:
                    print("Generating heatmaps and distributions")
                metrics.plot_metrics(densities_ensemble, densities_ref, weights, t + 1)

    except KeyboardInterrupt:
        print("Asked for interruption!")

    print("Saving everything now...\nDo not interrupt.")

    ds_densities_ensemble = nc.Dataset(datapaths.ds_densities_ensemble, "r+")
    ds_densities_ensemble["density"][:, :, :, :] = densities_ensemble
    ds_densities_ensemble.close()

    ds_parts_ensembles = nc.Dataset(datapaths.ds_parts_ensemble, "r+")
    ds_parts_ensembles["weight"][:, :] = weights
    ds_parts_ensembles.close()

    plt.close("all")
    metrics.csv_logger.export_csv()
