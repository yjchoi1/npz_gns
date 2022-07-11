import pickle
import numpy as np

# Inputs
def pkl2npz(ids=list, particle_type=int, save_name=str):
    """

    :param ids:
    :param particle_type:
    :param save_name:
    :return:
    """
    trajectories = {}
    for id in ids:
        with open(f'./pkl_data/rollout_{id}.pkl', 'rb') as f:
            data = pickle.load(f)
        positions = np.concatenate(
            (data["initial_positions"], data["ground_truth_rollout"])
        )

        trajectories[f"simulation_trajectory_{id}"] = (
            positions, np.full(positions.shape[1], particle_type, dtype=int)
        )

    np.savez_compressed(save_name, **trajectories)

def see_npz(file_name, trajectory_id):
    npz = np.load(f"{file_name}", allow_pickle=True)
    iter = (i for i in npz)
    ntrjec = sum(1 for _ in iter)
    print(f"number of trajectories: {ntrjec}")
    print("Length of .npz:", len(npz[f"simulation_trajectory_{trajectory_id}"]))
    print(f"Shape of particle positions for trajectory{trajectory_id}:", npz[f"simulation_trajectory_{trajectory_id}"][0].shape)
    print(f"Shape of particle types for trajectory{trajectory_id}", npz[f"simulation_trajectory_{trajectory_id}"][1].shape)

#%% Test

ids = [0, 1, 2]
particle_type = 6
save_name = 'test_pkl2npz.npz'
pkl2npz(ids, particle_type, save_name)

see_npz(file_name="test_pkl2npz.npz", trajectory_id=1)
