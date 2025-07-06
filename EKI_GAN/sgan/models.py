import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SimpleOSMMapEncoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, **kwargs):
        super().__init__()
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.max_elements = 10
        self.radius = 15

        self.element_encoder = nn.Linear(3, h_dim)

    def forward(self, trajectory_positions, map_elements):
        batch_size = trajectory_positions.size(1)
        last_pos = trajectory_positions[-1]

        batch_features = []
        for b in range(batch_size):
            nearest = self.find_nearest_elements(last_pos[b], map_elements)
            if nearest:
                features = torch.tensor([
                    sum(e['type'].item() for e in nearest) / len(nearest),
                    sum(e['distance'] for e in nearest) / len(nearest),
                    len(nearest)
                ], dtype=torch.float32).cuda()
                encoded = self.element_encoder(features)
            else:
                encoded = torch.zeros(self.h_dim).cuda()
            batch_features.append(encoded)

        result = torch.stack(batch_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return result

    def find_nearest_elements(self, position, map_elements):
        if not map_elements:
            return []

        pos_np = position.detach().cpu().numpy()
        distances_and_elements = []

        for element in map_elements:
            element_pos = np.array([
                element['geometry'][4].item(),
                element['geometry'][5].item()
            ])

            distance = np.linalg.norm(pos_np - element_pos)

            if distance <= self.radius:
                element_with_distance = {
                    'type': element['type'],
                    'distance': distance,
                    'position': element_pos
                }
                distances_and_elements.append((distance, element_with_distance))

        distances_and_elements.sort(key=lambda x: x[0])
        nearest = [elem for _, elem in distances_and_elements[:self.max_elements]]

        return nearest


class OSMMapProcessor:
    """OSM地图数据预处理器"""

    def __init__(self, osm_file_path):
        self.osm_file_path = osm_file_path
        self.nodes = {}
        self.ways = {}
        self.relations = {}
        self.processed_elements = []

    def parse_osm(self):
        tree = ET.parse(self.osm_file_path)
        root = tree.getroot()

        for node in root.findall('node'):
            node_id = node.get('id')
            lat = float(node.get('lat', 0))
            lon = float(node.get('lon', 0))
            self.nodes[node_id] = {'lat': lat, 'lon': lon}

        for way in root.findall('way'):
            way_id = way.get('id')
            way_data = {
                'nodes': [nd.get('ref') for nd in way.findall('nd')],
                'tags': {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
            }
            self.ways[way_id] = way_data

        for relation in root.findall('relation'):
            rel_id = relation.get('id')
            rel_data = {
                'members': [(member.get('type'), member.get('ref'), member.get('role', ''))
                            for member in relation.findall('member')],
                'tags': {tag.get('k'): tag.get('v') for tag in relation.findall('tag')}
            }
            self.relations[rel_id] = rel_data

    def process_map_elements(self):
        self.processed_elements = []

        type_mapping = {
            'virtual': 0, 'curbstone': 1, 'line_thin': 2, 'stop_line': 3,
            'traffic_light': 4, 'zebra_marking': 5, 'lanelet': 6, 'unknown': 7
        }

        for way_id, way_data in self.ways.items():
            tags = way_data['tags']
            way_type = tags.get('type', 'unknown')

            if way_type in type_mapping:
                geometry_features = self.calculate_geometry_features(way_data)
                semantic_features = self.calculate_semantic_features(tags)

                element = {
                    'type': torch.tensor(type_mapping[way_type], dtype=torch.long),
                    'geometry': torch.tensor(geometry_features, dtype=torch.float32),
                    'semantic': torch.tensor(semantic_features, dtype=torch.float32)
                }

                self.processed_elements.append(element)

        return self.processed_elements

    def calculate_geometry_features(self, way_data):
        node_refs = way_data['nodes']

        if len(node_refs) < 2:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        coords = []
        for node_ref in node_refs:
            if node_ref in self.nodes:
                node = self.nodes[node_ref]
                coords.append([node['lon'], node['lat']])

        if len(coords) < 2:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        coords = np.array(coords)

        total_length = 0
        for i in range(len(coords) - 1):
            total_length += np.linalg.norm(coords[i + 1] - coords[i])

        center = np.mean(coords, axis=0)

        if len(coords) >= 2:
            direction = coords[-1] - coords[0]
            angle = np.arctan2(direction[1], direction[0])
        else:
            angle = 0.0

        width = 3.5

        return [0.0, angle, width, total_length, center[0], center[1]]

    def calculate_semantic_features(self, tags):
        one_way = 1.0 if tags.get('one_way') == 'yes' else 0.0
        speed_limit = 50.0
        priority_mapping = {
            'stop_line': 1.0, 'traffic_light': 0.8, 'zebra_marking': 0.6,
            'virtual': 0.4, 'line_thin': 0.2
        }
        priority = priority_mapping.get(tags.get('type'), 0.5)
        connectivity = 1.0

        return [one_way, speed_limit / 100.0, priority, connectivity]


CUDA_LAUNCH_BLOCKING = 1


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0.0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """编码器，用于轨迹生成器和判别器"""

    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        """初始化LSTM的隐藏状态"""
        h_0 = torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        c_0 = torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        return (h_0, c_0)

    def forward(self, obs_traj):
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class PedestrianEncoder(nn.Module):
    """行人编码器 - 处理周围行人的轨迹和状态信息"""

    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, max_pedestrians=5, dropout=0.0):
        super(PedestrianEncoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.max_pedestrians = max_pedestrians

        self.pedestrian_spatial_embedding = nn.Linear(2, embedding_dim)
        self.pedestrian_trajectory_encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        self.pedestrian_state_embedding = nn.Linear(4, embedding_dim)  # vx, vy, ax, ay
        self.pedestrian_state_encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        self.attention_mlp = make_mlp([h_dim * 2, h_dim, 1], activation='relu', batch_norm=False, dropout=dropout)

        self.fusion_mlp = make_mlp([h_dim * 2, h_dim, h_dim], activation='relu', batch_norm=False, dropout=dropout)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, ped_obs_traj, ped_obs_traj_rel, ped_vx, ped_vy, ped_ax, ped_ay, ped_mask):
        """

        - ped_obs_traj: (obs_len, batch, 2, max_pedestrians) - 行人绝对轨迹
        - ped_obs_traj_rel: (obs_len, batch, 2, max_pedestrians) - 行人相对轨迹
        - ped_vx, ped_vy, ped_ax, ped_ay: (obs_len, batch, 1, max_pedestrians) - 行人状态
        - ped_mask: (obs_len, batch, 1, max_pedestrians) - 有效性掩码


        - pedestrian_encoding: (num_layers, batch, h_dim) - 行人编码特征
        """
        obs_len, batch_size, _, max_pedestrians = ped_obs_traj.shape


        pedestrian_encodings = []

        for b in range(batch_size):
            pedestrian_features = []

            for p in range(max_pedestrians):
                
                if ped_mask[:, b, :, p].sum() > 0:
                    # 提取当前行人的轨迹数据
                    curr_ped_traj_rel = ped_obs_traj_rel[:, b, :, p]  # (obs_len, 2)
                    curr_ped_state = torch.cat([
                        ped_vx[:, b, :, p],
                        ped_vy[:, b, :, p],
                        ped_ax[:, b, :, p],
                        ped_ay[:, b, :, p]
                    ], dim=1)  # (obs_len, 4)

                    curr_mask = ped_mask[:, b, 0, p]  # (obs_len,)

                    valid_indices = curr_mask > 0
                    if valid_indices.sum() >= 2:  # 至少需要2个有效时间步
                        valid_traj = curr_ped_traj_rel[valid_indices]
                        valid_state = curr_ped_state[valid_indices]

                        traj_embedding = self.pedestrian_spatial_embedding(valid_traj)
                        traj_embedding = traj_embedding.unsqueeze(1)  # (valid_len, 1, embedding_dim)
                        state_tuple = self.init_hidden(1)
                        _, traj_state = self.pedestrian_trajectory_encoder(traj_embedding, state_tuple)
                        traj_features = traj_state[0]  # (num_layers, 1, h_dim)
                        traj_features = traj_features.squeeze(1)  # (num_layers, h_dim)

 
                        state_embedding = self.pedestrian_state_embedding(valid_state)
                        state_embedding = state_embedding.unsqueeze(1)  # (valid_len, 1, embedding_dim)
                        state_tuple = self.init_hidden(1)
                        _, state_state = self.pedestrian_state_encoder(state_embedding, state_tuple)
                        state_features = state_state[0]  # (num_layers, 1, h_dim)
                        state_features = state_features.squeeze(1)  # (num_layers, h_dim)

                        if traj_features.dim() == 1:
                            traj_features = traj_features.unsqueeze(0)
                        if state_features.dim() == 1:
                            state_features = state_features.unsqueeze(0)

                        if traj_features.size(0) != self.num_layers:
                            if traj_features.size(0) == 1 and self.num_layers > 1:
                                traj_features = traj_features.repeat(self.num_layers, 1)
                            else:
                                traj_features = traj_features[:self.num_layers]

                        if state_features.size(0) != self.num_layers:
                            if state_features.size(0) == 1 and self.num_layers > 1:
                                state_features = state_features.repeat(self.num_layers, 1)
                            else:
                                state_features = state_features[:self.num_layers]

                        combined_features = torch.cat([traj_features, state_features], dim=1)  # (num_layers, h_dim*2)

                        batch_size_for_mlp = combined_features.size(0)  
                        reshaped_features = combined_features.view(-1, self.h_dim * 2)  # (batch_size_for_mlp, h_dim*2)
                        fused_features = self.fusion_mlp(reshaped_features)  # (batch_size_for_mlp, h_dim)

  
                        if fused_features.size(0) == self.num_layers:

                            final_features = fused_features
                        elif fused_features.size(0) > self.num_layers:

                            final_features = fused_features[-self.num_layers:]
                        else:

                            if fused_features.size(0) == 1:
                                final_features = fused_features.repeat(self.num_layers, 1)
                            else:
                                avg_features = torch.mean(fused_features, dim=0, keepdim=True)
                                final_features = avg_features.repeat(self.num_layers, 1)

                        pedestrian_features.append(final_features)  # (num_layers, h_dim)

            if not pedestrian_features:
                batch_pedestrian_encoding = torch.zeros(self.num_layers, self.h_dim).cuda()
            elif len(pedestrian_features) == 1:
                batch_pedestrian_encoding = pedestrian_features[0]
            else:
                pedestrian_stack = torch.stack(pedestrian_features, dim=0)  # (num_peds, num_layers, h_dim)
                num_peds, num_layers, h_dim = pedestrian_stack.shape

                last_layer_features = pedestrian_stack[:, -1, :]  # (num_peds, h_dim)

                attention_input = last_layer_features.view(num_peds, -1)
                mean_feature = torch.mean(last_layer_features, dim=0, keepdim=True).repeat(num_peds, 1)

                if num_peds > 1:
                    attention_concat = torch.cat([attention_input, mean_feature], dim=1)
                    attention_scores = self.attention_mlp(attention_concat)
                    attention_weights = torch.softmax(attention_scores, dim=0)  # (num_peds, 1)
                else:
                    attention_weights = torch.ones(1, 1).cuda()

                weighted_features = pedestrian_stack * attention_weights.unsqueeze(1).unsqueeze(2)
                batch_pedestrian_encoding = torch.sum(weighted_features, dim=0)  # (num_layers, h_dim)

            if batch_pedestrian_encoding.dim() != 2:
                batch_pedestrian_encoding = batch_pedestrian_encoding.view(-1, self.h_dim)
                if batch_pedestrian_encoding.size(0) != self.num_layers:
                    if batch_pedestrian_encoding.size(0) > self.num_layers:
                        batch_pedestrian_encoding = batch_pedestrian_encoding[-self.num_layers:]
                    else:
                        padding_size = self.num_layers - batch_pedestrian_encoding.size(0)
                        padding = torch.zeros(padding_size, self.h_dim).cuda()
                        batch_pedestrian_encoding = torch.cat([batch_pedestrian_encoding, padding], dim=0)
            elif batch_pedestrian_encoding.size(0) != self.num_layers or batch_pedestrian_encoding.size(
                    1) != self.h_dim:
                if batch_pedestrian_encoding.numel() == self.num_layers * self.h_dim:
                    batch_pedestrian_encoding = batch_pedestrian_encoding.view(self.num_layers, self.h_dim)
                else:
                    batch_pedestrian_encoding = torch.zeros(self.num_layers, self.h_dim).cuda()

            pedestrian_encodings.append(batch_pedestrian_encoding)

        for i, encoding in enumerate(pedestrian_encodings):
            if encoding.shape != (self.num_layers, self.h_dim):
                print(
                    f"Warning: Pedestrian encoding {i} has shape {encoding.shape}, expected ({self.num_layers}, {self.h_dim})")
                print(f"Tensor size: {encoding.numel()}, expected size: {self.num_layers * self.h_dim}")
                if encoding.numel() == self.num_layers * self.h_dim:
                    pedestrian_encodings[i] = encoding.view(self.num_layers, self.h_dim)
                else:
                    pedestrian_encodings[i] = torch.zeros(self.num_layers, self.h_dim).cuda()

        final_encoding = torch.stack(pedestrian_encodings, dim=1)  # (num_layers, batch, h_dim)

        return final_encoding


class TrafficEncoder(nn.Module):
    def __init__(self, traffic_state_dim=5, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(TrafficEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.traffic_embedding = nn.Embedding(traffic_state_dim, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, traffic_state):
        traffic_state = traffic_state.long()
        batch_size = traffic_state.size(1)
        seq_len = traffic_state.size(0)

        flat_traffic_state = traffic_state.reshape(-1)
        traffic_state_embedding = self.traffic_embedding(flat_traffic_state)
        traffic_state_embedding = traffic_state_embedding.view(seq_len, batch_size, -1)
        state_tuple = self.init_hidden(batch_size)
        output, state = self.encoder(traffic_state_embedding, state_tuple)
        final_h = state[0]
        return final_h


class VehicleEncoder(nn.Module):
    def __init__(self, agent_type_dim=6, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(VehicleEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.agent_type_embedding = nn.Embedding(agent_type_dim, embedding_dim)
        self.size_layer = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(2 * embedding_dim, h_dim, num_layers, dropout=dropout)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, agent_type, size):
        agent_type = agent_type.long()
        batch_size = agent_type.size(1)

        agent_type_embedding = self.agent_type_embedding(agent_type)
        agent_type_embedding = torch.squeeze(agent_type_embedding, -2)
        size_embedding = self.size_layer(size)

        combined_embedding = torch.cat([agent_type_embedding, size_embedding], dim=-1).view(-1, batch_size,
                                                                                            self.embedding_dim * 2)
        state_tuple = self.init_hidden(batch_size)
        output, state = self.encoder(combined_embedding, state_tuple)
        final_h = state[0]
        return final_h


class StateEncoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(StateEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.state_layer = nn.Linear(4, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, state):
        batch = state.size(1)
        state_embedding = self.state_layer(state)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(state_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):

    def __init__(
            self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
            pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
            activation='relu', batch_norm=True, pooling_type='pool_net',
            neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'atten_net':
                self.pool_net = StableEnhancedAttenPoolNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, vx, vy):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos, vx, vy)
                decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos
            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """池化模块"""

    def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
            activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class AttenPoolNet(PoolHiddenNet):
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
                 activation='relu', batch_norm=True, dropout=0.0):
        super(AttenPoolNet, self).__init__(embedding_dim, h_dim, mlp_dim, bottleneck_dim,
                                           activation, batch_norm, dropout)

        self.velocity_embedding = nn.Linear(2, embedding_dim)
        self.attention_mlp = make_mlp(
            [embedding_dim * 2, mlp_dim, 1],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def compute_attention_weights(self, rel_pos_embedding, velocity_embedding):
        concatenated = torch.cat([rel_pos_embedding, velocity_embedding], dim=1)
        attention_scores = self.attention_mlp(concatenated)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights

    def forward(self, h_states, seq_start_end, end_pos, vx, vy):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start

            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]

            curr_hidden_repeated = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_repeated = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_transposed = curr_end_pos.repeat(1, num_ped).view(num_ped * num_ped, -1)
            curr_rel_pos = curr_end_pos_repeated - curr_end_pos_transposed
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)

            curr_vx = vx[-1, start:end].repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_vy = vy[-1, start:end].repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_velocity = torch.cat((curr_vx, curr_vy), dim=1)
            curr_velocity_embedding = self.velocity_embedding(curr_velocity)

            attention_weights = self.compute_attention_weights(curr_rel_embedding, curr_velocity_embedding)

            weighted_h_input = torch.cat([curr_rel_embedding, curr_hidden_repeated], dim=1)
            weighted_h_input *= 0.05 * attention_weights.view(-1, 1)

            curr_pool_h = self.mlp_pre_pool(weighted_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]

            pool_h.append(curr_pool_h)

        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class StableEnhancedAttenPoolNet(PoolHiddenNet):
    """稳定版增强注意力池化网络 - 修复NaN问题"""

    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
                 activation='relu', batch_norm=True, dropout=0.0):
        super(StableEnhancedAttenPoolNet, self).__init__(embedding_dim, h_dim, mlp_dim, bottleneck_dim,
                                                         activation, batch_norm, dropout)

        self.velocity_embedding = nn.Linear(2, embedding_dim)

        self.distance_embedding = nn.Linear(1, embedding_dim // 8)  # 减小到8分之一
        self.direction_embedding = nn.Linear(2, embedding_dim // 8)
        self.velocity_diff_embedding = nn.Linear(2, embedding_dim // 8)

        enhanced_input_dim = embedding_dim * 2 + (embedding_dim // 8) * 3
        self.enhanced_attention_mlp = nn.Sequential(
            nn.Linear(enhanced_input_dim, mlp_dim // 4),
            nn.LayerNorm(mlp_dim // 4),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 4, 1)
        )

        self.adaptive_gate = nn.Sequential(
            nn.Linear(h_dim, h_dim // 8),
            nn.LayerNorm(h_dim // 8),
            nn.ReLU(),
            nn.Linear(h_dim // 8, 1),
            nn.Sigmoid()
        )

        self.eps = 1e-8

    def compute_enhanced_features(self, rel_pos, velocity, hidden_states):
        """稳定的特征计算"""
        batch_size = rel_pos.size(0)

        distances = torch.norm(rel_pos + self.eps, dim=1, keepdim=True)
        distances = torch.clamp(distances, min=self.eps, max=50.0)  
        distance_features = self.distance_embedding(distances)

        angles = torch.atan2(rel_pos[:, 1:2], rel_pos[:, 0:1] + self.eps)
        direction_features = self.direction_embedding(
            torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
        )


        velocity_clamped = torch.clamp(velocity, min=-20.0, max=20.0)  # 限制速度范围
        velocity_diff_features = self.velocity_diff_embedding(velocity_clamped)

        return distance_features, direction_features, velocity_diff_features

    def compute_attention_weights(self, rel_pos_embedding, velocity_embedding,
                                  distance_features, direction_features, velocity_diff_features):

        concatenated = torch.cat([
            rel_pos_embedding,
            velocity_embedding,
            distance_features,
            direction_features,
            velocity_diff_features
        ], dim=1)

        attention_scores = self.enhanced_attention_mlp(concatenated)

        attention_scores = torch.clamp(attention_scores, min=-10, max=10)  # 限制logits范围
        attention_weights = torch.softmax(attention_scores, dim=0)

        attention_weights = attention_weights + self.eps
        attention_weights = attention_weights / attention_weights.sum(dim=0, keepdim=True)

        return attention_weights

    def forward(self, h_states, seq_start_end, end_pos, vx, vy, time_step=None):
        pool_h = []

        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start

            if num_ped <= 0:
                continue

            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]

            if num_ped == 1:
                single_input = torch.cat([
                    torch.zeros(1, self.embedding_dim).cuda(),  
                    curr_hidden
                ], dim=1)
                curr_pool_h = self.mlp_pre_pool(single_input)
                pool_h.append(curr_pool_h)
                continue

            curr_hidden_repeated = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_repeated = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_transposed = curr_end_pos.repeat(1, num_ped).view(num_ped * num_ped, -1)

 
            curr_rel_pos = curr_end_pos_repeated - curr_end_pos_transposed
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)

       
            curr_vx = vx[-1, start:end].repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_vy = vy[-1, start:end].repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_velocity = torch.cat((curr_vx, curr_vy), dim=1)
            curr_velocity_embedding = self.velocity_embedding(curr_velocity)

         
            distance_features, direction_features, velocity_diff_features = \
                self.compute_enhanced_features(curr_rel_pos, curr_velocity, curr_hidden_repeated)

     
            attention_weights = self.compute_attention_weights(
                curr_rel_embedding, curr_velocity_embedding,
                distance_features, direction_features, velocity_diff_features
            )

            gate_weights = self.adaptive_gate(curr_hidden_repeated)
            attention_weights = attention_weights * gate_weights

            weighted_h_input = torch.cat([curr_rel_embedding, curr_hidden_repeated], dim=1)
            weighted_h_input *= attention_weights.view(-1, 1)

            curr_pool_h = self.mlp_pre_pool(weighted_h_input)

            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]

            pool_h.append(curr_pool_h)

        if not pool_h:  
            return torch.zeros(1, self.bottleneck_dim).cuda()

        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class MultiScaleAttenPoolNet(PoolHiddenNet):

    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
                 activation='relu', batch_norm=True, dropout=0.0):
        super(MultiScaleAttenPoolNet, self).__init__(embedding_dim, h_dim, mlp_dim, bottleneck_dim,
                                                     activation, batch_norm, dropout)

        self.velocity_embedding = nn.Linear(2, embedding_dim)

        self.local_attention = make_mlp([embedding_dim * 2, mlp_dim // 4, 1],
                                        activation, batch_norm, dropout)
        self.global_attention = make_mlp([embedding_dim * 2, mlp_dim // 4, 1],
                                         activation, batch_norm, dropout)

        self.scale_fusion = nn.Linear(2, 1)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=h_dim,
            num_heads=2,  
            dropout=dropout,
            batch_first=True
        )

    def compute_multi_scale_attention(self, rel_pos_embedding, velocity_embedding, distances):

        concatenated = torch.cat([rel_pos_embedding, velocity_embedding], dim=1)


        local_weights = self.local_attention(concatenated)
        local_mask = (distances < 2.0).float().unsqueeze(-1)  
        local_weights = local_weights * local_mask


        global_weights = self.global_attention(concatenated)

        scale_weights = torch.cat([local_weights, global_weights], dim=1)
        fusion_weights = torch.softmax(self.scale_fusion(scale_weights), dim=0)

        final_weights = fusion_weights[:, 0:1] * local_weights + fusion_weights[:, 1:2] * global_weights

        return torch.softmax(final_weights, dim=0)

    def forward(self, h_states, seq_start_end, end_pos, vx, vy):
        pool_h = []

        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start

            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]

            curr_hidden_repeated = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_repeated = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_transposed = curr_end_pos.repeat(1, num_ped).view(num_ped * num_ped, -1)
            curr_rel_pos = curr_end_pos_repeated - curr_end_pos_transposed

            distances = torch.norm(curr_rel_pos, dim=1)

            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)

            curr_vx = vx[-1, start:end].repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_vy = vy[-1, start:end].repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_velocity = torch.cat((curr_vx, curr_vy), dim=1)
            curr_velocity_embedding = self.velocity_embedding(curr_velocity)

            attention_weights = self.compute_multi_scale_attention(
                curr_rel_embedding, curr_velocity_embedding, distances
            )

            if num_ped > 1:
                hidden_reshaped = curr_hidden.unsqueeze(0)  # (1, num_ped, h_dim)
                cross_attended, _ = self.cross_attention(
                    hidden_reshaped, hidden_reshaped, hidden_reshaped
                )
                cross_attended = cross_attended.squeeze(0).repeat(num_ped, 1)
            else:
                cross_attended = curr_hidden_repeated

            enhanced_hidden = 0.8 * curr_hidden_repeated + 0.2 * cross_attended

            weighted_h_input = torch.cat([curr_rel_embedding, enhanced_hidden], dim=1)
            weighted_h_input *= attention_weights.view(-1, 1)

            curr_pool_h = self.mlp_pre_pool(weighted_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]

            pool_h.append(curr_pool_h)

        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class SocialPooling(nn.Module):

    def __init__(
            self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
            neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                top_left, curr_end_pos).type_as(seq_start_end)
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1
            within_bound = within_bound.view(-1)

            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos, curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class TrajectoryGenerator(nn.Module):
    def __init__(
            self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
            decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0,),
            noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
            pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
            activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8,
            traffic_h_dim=64, map_h_dim=32, pedestrian_h_dim=32, max_pedestrians=5
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.traffic_h_dim = traffic_h_dim
        self.map_h_dim = map_h_dim
        self.pedestrian_h_dim = pedestrian_h_dim

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = bottleneck_dim

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.traffic_encoder = TrafficEncoder(traffic_state_dim=5, embedding_dim=64, h_dim=traffic_h_dim)
        self.vehicle_encoder = VehicleEncoder(agent_type_dim=6, embedding_dim=64, h_dim=64)
        self.state_encoder = StateEncoder(embedding_dim=embedding_dim, h_dim=64)

        self.pedestrian_encoder = PedestrianEncoder(
            embedding_dim=embedding_dim,
            h_dim=pedestrian_h_dim,
            num_layers=num_layers,
            max_pedestrians=max_pedestrians,
            dropout=dropout
        )

        self.map_encoder = SimpleOSMMapEncoder(embedding_dim=embedding_dim, h_dim=map_h_dim)

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        elif pooling_type == 'atten_net':
            self.pool_net = StableEnhancedAttenPoolNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                activation=activation,
                bottleneck_dim=bottleneck_dim,
                batch_norm=batch_norm,
                dropout=dropout
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        if pooling_type:
            input_dim = encoder_h_dim * 3 + bottleneck_dim + traffic_h_dim + map_h_dim + pedestrian_h_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)
        return decoder_h

    def mlp_decoder_needed(self):
        if (
                self.noise_dim or self.pooling_type or
                self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, vx, vy, ax, ay,
                agent_type, size, traffic_state, map_elements=None,
                ped_obs_traj=None, ped_obs_traj_rel=None, ped_vx=None, ped_vy=None,
                ped_ax=None, ped_ay=None, ped_mask=None, user_noise=None):

        batch = obs_traj_rel.size(1)

        final_encoder_h = self.encoder(obs_traj_rel)
        vehicle_encoding = self.vehicle_encoder(agent_type, size)
        state_encoding = self.state_encoder(torch.cat([vx, vy, ax, ay], dim=2))
        traffic_encoding = self.traffic_encoder(traffic_state)

        if map_elements is not None:
            map_encoding = self.map_encoder(obs_traj, map_elements)
        else:
            map_encoding = torch.zeros(self.num_layers, batch, self.map_h_dim).cuda()

        if (ped_obs_traj is not None and ped_obs_traj_rel is not None and
                ped_vx is not None and ped_vy is not None and
                ped_ax is not None and ped_ay is not None and ped_mask is not None):
            pedestrian_encoding = self.pedestrian_encoder(
                ped_obs_traj, ped_obs_traj_rel, ped_vx, ped_vy, ped_ax, ped_ay, ped_mask
            )
        else:
            pedestrian_encoding = torch.zeros(self.num_layers, batch, self.pedestrian_h_dim).cuda()

        combined_encoding = torch.cat([final_encoder_h, vehicle_encoding, state_encoding], dim=2)

        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(combined_encoding, seq_start_end, end_pos, vx, vy)

            mlp_decoder_context_input = torch.cat([
                combined_encoding.view(-1, self.encoder_h_dim * 3),
                pool_h,
                traffic_encoding.view(-1, self.traffic_h_dim),
                map_encoding.view(-1, self.map_h_dim),
                pedestrian_encoding.view(-1, self.pedestrian_h_dim)  # 新增行人编码
            ], dim=1)
        else:
            mlp_decoder_context_input = combined_encoding.view(-1, self.encoder_h_dim)

        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input

        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        decoder_out = self.decoder(
            last_pos, last_pos_rel, state_tuple, seq_start_end, vx, vy
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(
            self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
            num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
            d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        final_h = self.encoder(traj_rel)
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores
