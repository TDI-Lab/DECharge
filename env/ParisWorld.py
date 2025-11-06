import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree


class ParisWorld:

    def __init__(self, grids, vehicles, data_read_num,
                 time_style='weekend', grid_style='square',
                 avail_prob=1, neighbor_num=1):

        data_path_parent = os.path.join(os.getcwd(), f'datasets/')
        important_features = ['dim_id', 'grid_id', 'x', 'y', 'slot_num', 'travel', 'cost', 'plan']

        # 1. Method to distribute spatio-temporal EVs: self-generate a distribution
        data_generated = []
        for i in range(data_read_num):
            path = data_path_parent + f'ParisChargingDemands_{time_style}/{vehicles}EVs/demands_dist{i}.csv'
            df = pd.read_csv(path)
            data_generated.append(df)
        self.data_train = data_generated[: int(data_read_num * 0.7)]
        self.data_test = data_generated[int(data_read_num * 0.7):]

        # 2. Set features to station, including x, y
        self.station_data = pd.read_csv(data_path_parent + f'info_static_{avail_prob}.csv')
        # 2.1. previous, merge all slots in the same station and counts the number of slots
        slots_arr = self.station_data.groupby('s_id', as_index=False)['Availability'].sum()['Availability'].to_numpy()
        self.station_data = self.station_data.drop_duplicates(subset=['s_id'])
        self.station_data['slot_num'] = slots_arr

        # 2.2. Transform to meters
        latitudes = self.station_data['latitude'].to_list()
        longitudes = self.station_data['longitude'].to_list()
        # Define the upper-left corner (reference point)
        upper_left = (48.815, 2.255)  # Example: (latitude, longitude)
        # Earth's approximate values
        meters_per_deg_lat = 111320  # Meters per degree latitude
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(upper_left[0]))  # Adjust for longitude
        # Convert lat/lon to relative positions based on the upper-left corner
        self.station_data['x'] = (np.array(longitudes) - upper_left[1]) * meters_per_deg_lon  # Longitude difference
        self.station_data['y'] = (np.array(latitudes) - upper_left[
            0]) * meters_per_deg_lat  # Latitude difference (inverted for plotting)

        # 2.3. Set grids nad neighbouring idx: square or voronoi
        self.grids = grids
        # add params to the stations dataset
        if grid_style == 'square':
            grid_neighbours = self.add_grid_indexes_square(neighbour_search_num=neighbor_num)
        elif grid_style == 'voronoi':
            self.voronoi_tree = None
            grid_neighbours = self.add_grid_indexes_voronoi()
        else:
            print("Wrong grid division style!!!")
            return

        # 2.4. Add dimension index
        stations_num = len(self.station_data)
        self.station_data = self.station_data.sort_values(by=['grid_id', 's_id'])
        self.station_data['dim_id'] = range(stations_num)

        # 2.5. Add the cost column and plan column
        self.station_data['cost'] = np.zeros(stations_num)
        self.station_data['travel'] = np.zeros(stations_num)
        plans = np.eye(stations_num)  # create an identity matrix where each row is a one-hot vector
        self.station_data['plan'] = list(plans)
        self.station_data = self.station_data.reset_index()

        # # 3. Find station info for each grid and its neighbours
        # self.grid_with_neighbors = []
        # grid_groups = self.station_data.groupby('grid_id')
        # grid_dict = {grid_id: group[important_features].values for grid_id, group in grid_groups}
        # # Find stations in each grid and its 8 neighboring grids
        # for grid_id in range(grids):
        #     neighbors = grid_neighbours[grid_id]
        #     neighbor_points = np.vstack([grid_dict[n] for n in neighbors if n in grid_dict]) \
        #         if neighbors else np.array([])
        #     # Transform back to dataframe
        #     points_df = pd.DataFrame(columns=important_features)
        #     points_df[important_features] = np.vstack((grid_dict[grid_id], neighbor_points)) \
        #         if grid_id in grid_dict else neighbor_points
        #     points_df[['dim_id', 'grid_id']] = points_df[['dim_id', 'grid_id']].astype(int)
        #     points_df = points_df.reset_index(drop=True)
        #     # Store points in dictionary
        #     self.grid_with_neighbors.append(points_df)

        self.grid_with_neighbors = []
        self.station_data['dist_to_center'] = np.zeros(stations_num)
        edge_num = np.sqrt(self.grids)
        x_step, y_step = 12000 / edge_num, 10000 / edge_num
        search_range = neighbor_num * 3600
        for grid_id in range(grids):
            center_x = grid_id % edge_num * x_step + x_step / 2
            center_y = grid_id // edge_num * y_step + y_step / 2
            self.station_data['dist_to_center'] = np.sqrt((self.station_data['x'] - center_x) ** 2 +
                                                          (self.station_data['y'] - center_y) ** 2)
            points_df = self.station_data[self.station_data['dist_to_center'] < search_range][important_features].copy()
            points_df[['dim_id', 'grid_id']] = points_df[['dim_id', 'grid_id']].astype(int)
            points_df = points_df.reset_index(drop=True)
            self.grid_with_neighbors.append(points_df)

    def add_grid_indexes_square(self, neighbour_search_num=1):
        n = neighbour_search_num
        # set grid index
        num_rows, num_cols = np.sqrt(self.grids), np.sqrt(self.grids)
        x_step, y_step = 12000 / num_cols, 10000 / num_rows
        # Assign grid index to each point
        self.station_data['grid_x'] = (self.station_data['x'] // x_step).astype(int)
        self.station_data['grid_y'] = (self.station_data['y'] // y_step).astype(int)
        self.station_data['grid_id'] = self.station_data['grid_x'] + self.station_data['grid_y'] * num_cols
        # add neighbouring grid indexes
        grid_neighbours = {}
        for grid_id in range(self.grids):
            grid_x, grid_y = grid_id % num_cols, grid_id // num_cols  # Convert back to (x, y)

            neighbors = []
            for dx in range(-n, n+1):
                for dy in (-n, n+1):
                    if dx == 0 and dy == 0:
                        continue  # Skip the current grid
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < num_cols and 0 <= ny < num_rows:
                        neighbors.append(nx + ny * num_cols)
            grid_neighbours[grid_id] = neighbors

        return grid_neighbours

    def add_grid_indexes_voronoi(self):
        # Generate clustering grids
        kmeans = KMeans(n_clusters=self.grids, random_state=0)
        kmeans.fit(self.station_data[['x', 'y']])
        # Get the cluster centroids (these are the seeds for the Voronoi diagram),
        # and sort them from bottom left to upper right
        voronoi_sites = kmeans.cluster_centers_
        voronoi_sites = voronoi_sites[np.lexsort((voronoi_sites[:, 1], voronoi_sites[:, 0]))]
        voronoi = Voronoi(voronoi_sites)
        self.voronoi_tree = cKDTree(voronoi_sites)

        # Assign each station to a nearest voronoi region
        points = self.station_data[['x', 'y']].to_numpy()[:, None]
        labels = np.argmin(np.linalg.norm(points - voronoi_sites, axis=2), axis=1)
        self.station_data['grid_id'] = labels
        # Find neighboring grids for each point
        grid_neighbours = {i: [] for i in range(len(voronoi_sites))}
        for ridge in voronoi.ridge_points:
            p1, p2 = ridge
            grid_neighbours[p1].append(p2)
            grid_neighbours[p2].append(p1)
        return grid_neighbours

    def find_square_index(self, data):
        data = data.copy()
        num_rows, num_cols = np.sqrt(self.grids), np.sqrt(self.grids)
        x_step, y_step = 12000 / num_cols, 10000 / num_rows
        square_indexes = (data['x'] // x_step).astype(int) + (data['y'] // y_step).astype(int) * num_cols
        return square_indexes

    def find_voronoi_index(self, data):
        _, voronoi_indices = self.voronoi_tree.query(data)
        return voronoi_indices
