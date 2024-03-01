import numpy as np
from typing import List
import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap


def plot_single_vehicle(
    avm: ArgoverseMap, 
    sample_past_trajectory: np.ndarray,  # (20, 2)
    sample_groundtruth: np.ndarray,  # (20, 2) 
    sample_forecasted_trajectories: List[np.ndarray],  # List[(30, 2)]
    sample_city_name: str, 
    ls: str):

    sample_groundtruth = np.concatenate((np.expand_dims(sample_past_trajectory[-1], axis=0), sample_groundtruth), axis=0)
    for sample_forecasted_trajectory in sample_forecasted_trajectories:
        sample_forecasted_trajectory = np.concatenate((np.expand_dims(sample_past_trajectory[-1], axis=0), sample_forecasted_trajectory), axis=0)

    ## Plot history
    obs_len = sample_past_trajectory.shape[0]
    pred_len = sample_groundtruth.shape[0]
    plt.plot(
        sample_past_trajectory[:, 0],
        sample_past_trajectory[:, 1],
        color="#ECA154",
        label="Past Trajectory",
        alpha=1,
        linewidth=3,
        zorder=15,
        ls = ls
    )

    ## Plot future
    plt.plot(
        sample_groundtruth[:, 0],
        sample_groundtruth[:, 1],
        color="#d33e4c",
        label="Ground Truth",
        alpha=1,
        linewidth=3.0,
        zorder=20,
        ls = "--"
    )

    ## Plot prediction
    for j in range(len(sample_forecasted_trajectories)):
        plt.plot(
            sample_forecasted_trajectories[j][:, 0],
            sample_forecasted_trajectories[j][:, 1],
            color="#007672",
            label="Forecasted Trajectory",
            alpha=1,
            linewidth=3.2,
            zorder=15,
            ls = "--"
        )
        
        # Plot the end marker for forcasted trajectories
        plt.arrow(
            sample_forecasted_trajectories[j][-2, 0], 
            sample_forecasted_trajectories[j][-2, 1],
            sample_forecasted_trajectories[j][-1, 0] - sample_forecasted_trajectories[j][-2, 0],
            sample_forecasted_trajectories[j][-1, 1] - sample_forecasted_trajectories[j][-2, 1],
            color="#007672",
            label="Forecasted Trajectory",
            alpha=1,
            linewidth=3.2,
            zorder=15,
            head_width=1.1,
        )
    
    ## Plot the end marker for history
    plt.arrow(
            sample_past_trajectory[-2, 0], 
            sample_past_trajectory[-2, 1],
            sample_past_trajectory[-1, 0] - sample_past_trajectory[-2, 0],
            sample_past_trajectory[-1, 1] - sample_past_trajectory[-2, 1],
            color="#ECA154",
            label="Past Trajectory",
            alpha=1,
            linewidth=3,
            zorder=25,
            head_width=1.0,
        )

    ## Plot the end marker for future
    plt.arrow(
            sample_groundtruth[-2, 0], 
            sample_groundtruth[-2, 1],
            sample_groundtruth[-1, 0] - sample_groundtruth[-2, 0],
            sample_groundtruth[-1, 1] - sample_groundtruth[-2, 1],
            color="#d33e4c",
            label="Ground Truth",
            alpha=1,
            linewidth=3.0,
            zorder=25,
            head_width=1.0,
        )

    ## Plot history context
    for j in range(obs_len):
        lane_ids = avm.get_lane_ids_in_xy_bbox(
            sample_past_trajectory[j, 0],
            sample_past_trajectory[j, 1],
            sample_city_name,
            query_search_range_manhattan=65,
        )
        [avm.draw_lane(lane_id, sample_city_name) for lane_id in lane_ids]

    ## Plot future context
    for j in range(pred_len):
        lane_ids = avm.get_lane_ids_in_xy_bbox(
            sample_groundtruth[j, 0],
            sample_groundtruth[j, 1],
            sample_city_name,
            query_search_range_manhattan=65,
        )
        [avm.draw_lane(lane_id, sample_city_name) for lane_id in lane_ids]