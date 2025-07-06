import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

_obs_len = 12
_pred_len = 18
logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list,
     vx_list, vy_list, ax_list, ay_list,
     agent_type_list, size_list, traffic_state_list,
     ped_obs_seq_list, ped_obs_seq_rel_list, ped_vx_list, ped_vy_list,
     ped_ax_list, ped_ay_list, ped_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)

    vx = torch.cat(vx_list, dim=0).permute(2, 0, 1)
    vy = torch.cat(vy_list, dim=0).permute(2, 0, 1)
    ax = torch.cat(ax_list, dim=0).permute(2, 0, 1)
    ay = torch.cat(ay_list, dim=0).permute(2, 0, 1)
    agent_type = torch.cat(agent_type_list, dim=0).permute(2, 0, 1)
    size = torch.cat(size_list, dim=0).permute(2, 0, 1)
    traffic_state = torch.cat(traffic_state_list, dim=0).permute(2, 0, 1)

    ped_obs_traj = torch.cat(ped_obs_seq_list, dim=0).permute(2, 0, 1, 3)
    ped_obs_traj_rel = torch.cat(ped_obs_seq_rel_list, dim=0).permute(2, 0, 1, 3)
    ped_vx = torch.cat(ped_vx_list, dim=0).permute(2, 0, 1, 3)
    ped_vy = torch.cat(ped_vy_list, dim=0).permute(2, 0, 1, 3)
    ped_ax = torch.cat(ped_ax_list, dim=0).permute(2, 0, 1, 3)
    ped_ay = torch.cat(ped_ay_list, dim=0).permute(2, 0, 1, 3)
    ped_mask = torch.cat(ped_mask_list, dim=0).permute(2, 0, 1, 3)

    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end,
        vx, vy, ax, ay, agent_type, size, traffic_state,
        ped_obs_traj, ped_obs_traj_rel, ped_vx, ped_vy, ped_ax, ped_ay, ped_mask
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '

    if not os.path.exists(_path):
        return np.array([])

    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            if len(line) == 1:
                line = line[0].split(',')
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def read_pedestrian_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '

    if not os.path.exists(_path):
        return np.array([])

    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            if len(line) == 1:
                line = line[0].split(',')
            if len(line) >= 8:
                reordered = [line[1], line[0], line[2], line[3], line[4], line[5], line[6], line[7]]
                line = [float(i) for i in reordered]
                data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):

    def __init__(
            self, data_dir, obs_len=_obs_len, pred_len=_pred_len, skip=15, threshold=30,
            min_ped=1, delim='\t', max_pedestrians=5  # 每个场景最多考虑的行人数量
    ):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.max_pedestrians = max_pedestrians

        vx_list = []
        vy_list = []
        ax_list = []
        ay_list = []
        agent_type_list = []
        size_list = []
        traffic_state_list = []

        ped_obs_seq_list = []
        ped_obs_seq_rel_list = []
        ped_vx_list = []
        ped_vy_list = []
        ped_ax_list = []
        ped_ay_list = []
        ped_mask_list = []

        veh_dir = os.path.join(self.data_dir, 'veh')
        ped_dir = os.path.join(self.data_dir, 'ped')

        if not os.path.exists(veh_dir):
            veh_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            veh_files = [os.path.join(self.data_dir, f) for f in veh_files]
        else:
            veh_files = [f for f in os.listdir(veh_dir) if f.endswith('.csv')]
            veh_files = [os.path.join(veh_dir, f) for f in veh_files]

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        for veh_path in veh_files:
            veh_data = read_file(veh_path, delim)
            if len(veh_data) == 0:
                continue

            veh_filename = os.path.basename(veh_path)
            ped_path = os.path.join(ped_dir, veh_filename) if os.path.exists(ped_dir) else None

            ped_data = np.array([])
            if ped_path and os.path.exists(ped_path):
                ped_data = read_pedestrian_file(ped_path, delim)

            veh_frames = np.unique(veh_data[:, 0]).tolist()
            veh_frame_data = []
            for frame in veh_frames:
                veh_frame_data.append(veh_data[frame == veh_data[:, 0], :])

            ped_frame_data = {}
            if len(ped_data) > 0:
                ped_frames = np.unique(ped_data[:, 0]).tolist()
                for frame in ped_frames:
                    ped_frame_data[frame] = ped_data[frame == ped_data[:, 0], :]

            num_sequences = int(math.ceil((len(veh_frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                if idx + self.seq_len > len(veh_frames):
                    break

                curr_seq_data = np.concatenate(veh_frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_vx = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
                curr_vy = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
                curr_ax = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
                curr_ay = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
                curr_size = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_agent_type = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
                curr_traffic_state = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))

                curr_ped_obs_seq = np.zeros((len(peds_in_curr_seq), 2, self.obs_len, self.max_pedestrians))
                curr_ped_obs_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.obs_len, self.max_pedestrians))
                curr_ped_vx = np.zeros((len(peds_in_curr_seq), 1, self.obs_len, self.max_pedestrians))
                curr_ped_vy = np.zeros((len(peds_in_curr_seq), 1, self.obs_len, self.max_pedestrians))
                curr_ped_ax = np.zeros((len(peds_in_curr_seq), 1, self.obs_len, self.max_pedestrians))
                curr_ped_ay = np.zeros((len(peds_in_curr_seq), 1, self.obs_len, self.max_pedestrians))
                curr_ped_mask = np.zeros((len(peds_in_curr_seq), 1, self.obs_len, self.max_pedestrians))

                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []

                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq_ = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq_ = np.around(curr_ped_seq_, decimals=4)

                    pad_front = veh_frames.index(curr_ped_seq_[0, 0]) - idx
                    pad_end = veh_frames.index(curr_ped_seq_[-1, 0]) - idx + 1

                    if pad_end - pad_front != self.seq_len:
                        continue

                    curr_ped_seq = np.transpose(curr_ped_seq_[:, 2:4])
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                    curr_vel_vx = np.transpose(curr_ped_seq_[:, 4])
                    curr_vel_vy = np.transpose(curr_ped_seq_[:, 5])
                    curr_vel_ax = np.transpose(curr_ped_seq_[:, 6])
                    curr_vel_ay = np.transpose(curr_ped_seq_[:, 7])
                    curr_vel_size = np.transpose(curr_ped_seq_[:, 8:10])
                    curr_vel_agent_type = np.transpose(curr_ped_seq_[:, 10])
                    curr_vel_traffic_state = np.transpose(curr_ped_seq_[:, 11])

                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_vx[_idx, :, pad_front:pad_end] = curr_vel_vx
                    curr_vy[_idx, :, pad_front:pad_end] = curr_vel_vy
                    curr_ax[_idx, :, pad_front:pad_end] = curr_vel_ax
                    curr_ay[_idx, :, pad_front:pad_end] = curr_vel_ay
                    curr_size[_idx, :, pad_front:pad_end] = curr_vel_size
                    curr_agent_type[_idx, :, pad_front:pad_end] = curr_vel_agent_type
                    curr_traffic_state[_idx, :, pad_front:pad_end] = curr_vel_traffic_state

                    self._process_pedestrian_data(
                        ped_frame_data, veh_frames, idx, pad_front, pad_end,
                        _idx, curr_ped_obs_seq, curr_ped_obs_seq_rel,
                        curr_ped_vx, curr_ped_vy, curr_ped_ax, curr_ped_ay, curr_ped_mask
                    )

                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

                    vx_list.append(curr_vx[:num_peds_considered])
                    vy_list.append(curr_vy[:num_peds_considered])
                    ax_list.append(curr_ax[:num_peds_considered])
                    ay_list.append(curr_ay[:num_peds_considered])
                    agent_type_list.append(curr_agent_type[:num_peds_considered])
                    size_list.append(curr_size[:num_peds_considered])
                    traffic_state_list.append(curr_traffic_state[:num_peds_considered])

                    ped_obs_seq_list.append(curr_ped_obs_seq[:num_peds_considered])
                    ped_obs_seq_rel_list.append(curr_ped_obs_seq_rel[:num_peds_considered])
                    ped_vx_list.append(curr_ped_vx[:num_peds_considered])
                    ped_vy_list.append(curr_ped_vy[:num_peds_considered])
                    ped_ax_list.append(curr_ped_ax[:num_peds_considered])
                    ped_ay_list.append(curr_ped_ay[:num_peds_considered])
                    ped_mask_list.append(curr_ped_mask[:num_peds_considered])

        self.num_seq = len(seq_list)

        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        vx_list = np.concatenate(vx_list, axis=0)
        vy_list = np.concatenate(vy_list, axis=0)
        ax_list = np.concatenate(ax_list, axis=0)
        ay_list = np.concatenate(ay_list, axis=0)
        agent_type_list = np.concatenate(agent_type_list, axis=0)
        size_list = np.concatenate(size_list, axis=0)
        traffic_state_list = np.concatenate(traffic_state_list, axis=0)

        ped_obs_seq_list = np.concatenate(ped_obs_seq_list, axis=0)
        ped_obs_seq_rel_list = np.concatenate(ped_obs_seq_rel_list, axis=0)
        ped_vx_list = np.concatenate(ped_vx_list, axis=0)
        ped_vy_list = np.concatenate(ped_vy_list, axis=0)
        ped_ax_list = np.concatenate(ped_ax_list, axis=0)
        ped_ay_list = np.concatenate(ped_ay_list, axis=0)
        ped_mask_list = np.concatenate(ped_mask_list, axis=0)

        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)

        self.vx = torch.from_numpy(vx_list[:, :, :self.obs_len]).type(torch.float)
        self.vy = torch.from_numpy(vy_list[:, :, :self.obs_len]).type(torch.float)
        self.ax = torch.from_numpy(ax_list[:, :, :self.obs_len]).type(torch.float)
        self.ay = torch.from_numpy(ay_list[:, :, :self.obs_len]).type(torch.float)
        self.size = torch.from_numpy(size_list[:, :, :self.obs_len]).type(torch.float)
        self.agent_type = torch.from_numpy(agent_type_list[:, :, :self.obs_len]).type(torch.float)
        self.traffic_state = torch.from_numpy(traffic_state_list[:, :, :self.obs_len]).type(torch.float)

        self.ped_obs_traj = torch.from_numpy(ped_obs_seq_list[:, :, :, :]).type(torch.float)
        self.ped_obs_traj_rel = torch.from_numpy(ped_obs_seq_rel_list[:, :, :, :]).type(torch.float)
        self.ped_vx = torch.from_numpy(ped_vx_list[:, :, :, :]).type(torch.float)
        self.ped_vy = torch.from_numpy(ped_vy_list[:, :, :, :]).type(torch.float)
        self.ped_ax = torch.from_numpy(ped_ax_list[:, :, :, :]).type(torch.float)
        self.ped_ay = torch.from_numpy(ped_ay_list[:, :, :, :]).type(torch.float)
        self.ped_mask = torch.from_numpy(ped_mask_list[:, :, :, :]).type(torch.float)

        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def _process_pedestrian_data(self, ped_frame_data, veh_frames, seq_idx, pad_front, pad_end,
                                 veh_idx, curr_ped_obs_seq, curr_ped_obs_seq_rel,
                                 curr_ped_vx, curr_ped_vy, curr_ped_ax, curr_ped_ay, curr_ped_mask):
        if not ped_frame_data:
            return

        obs_frames = veh_frames[seq_idx + pad_front:seq_idx + pad_front + self.obs_len]

        all_pedestrians = set()
        for frame_idx in obs_frames:
            if frame_idx in ped_frame_data:
                frame_peds = ped_frame_data[frame_idx]
                for ped_data in frame_peds:
                    all_pedestrians.add(int(ped_data[1]))  # ped_id

        pedestrians = list(all_pedestrians)[:self.max_pedestrians]

        for ped_slot, ped_id in enumerate(pedestrians):
            ped_positions = []
            ped_velocities = []
            ped_accelerations = []
            valid_frames = []

            for t, frame_idx in enumerate(obs_frames):
                if frame_idx in ped_frame_data:
                    frame_peds = ped_frame_data[frame_idx]
                    ped_found = False

                    for ped_data in frame_peds:
                        if int(ped_data[1]) == ped_id:  
                            ped_positions.append([ped_data[2], ped_data[3]])  # x, y
                            ped_velocities.append([ped_data[4], ped_data[5]])  # vx, vy
                            ped_accelerations.append([ped_data[6], ped_data[7]])  # ax, ay
                            valid_frames.append(t)
                            ped_found = True
                            break

                    if not ped_found:
                        ped_positions.append([0.0, 0.0])
                        ped_velocities.append([0.0, 0.0])
                        ped_accelerations.append([0.0, 0.0])
                else:
                    ped_positions.append([0.0, 0.0])
                    ped_velocities.append([0.0, 0.0])
                    ped_accelerations.append([0.0, 0.0])

            if len(valid_frames) >= 2:  
                ped_positions = np.array(ped_positions)
                ped_velocities = np.array(ped_velocities)
                ped_accelerations = np.array(ped_accelerations)

                ped_positions_rel = np.zeros_like(ped_positions)
                ped_positions_rel[1:] = ped_positions[1:] - ped_positions[:-1]

                curr_ped_obs_seq[veh_idx, :, :len(ped_positions), ped_slot] = ped_positions.T
                curr_ped_obs_seq_rel[veh_idx, :, :len(ped_positions_rel), ped_slot] = ped_positions_rel.T
                curr_ped_vx[veh_idx, :, :len(ped_velocities), ped_slot] = ped_velocities[:, 0:1].T
                curr_ped_vy[veh_idx, :, :len(ped_velocities), ped_slot] = ped_velocities[:, 1:2].T
                curr_ped_ax[veh_idx, :, :len(ped_accelerations), ped_slot] = ped_accelerations[:, 0:1].T
                curr_ped_ay[veh_idx, :, :len(ped_accelerations), ped_slot] = ped_accelerations[:, 1:2].T

                for t in valid_frames:
                    curr_ped_mask[veh_idx, :, t, ped_slot] = 1.0

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.vx[start:end, :], self.vy[start:end, :],
            self.ax[start:end, :], self.ay[start:end, :],
            self.agent_type[start:end, :], self.size[start:end, :],
            self.traffic_state[start:end, :],
            self.ped_obs_traj[start:end, :, :, :], self.ped_obs_traj_rel[start:end, :, :, :],
            self.ped_vx[start:end, :, :, :], self.ped_vy[start:end, :, :, :],
            self.ped_ax[start:end, :, :, :], self.ped_ay[start:end, :, :, :],
            self.ped_mask[start:end, :, :, :]
        ]
        return out