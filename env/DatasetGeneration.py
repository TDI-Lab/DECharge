import os
import shutil
import numpy as np
import pandas as pd


class DatasetGeneration:

    def __init__(self, grids, steps, data_num, num_points, world):
        self.grids = grids
        self.steps = steps
        self.data_num = data_num
        self.num_points = num_points
        self.world = world

        self.dataset_path = os.path.join(os.getcwd(), f'datasets/dataset_grids{grids}_steps{steps}_evs{num_points}/')
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)
        os.makedirs(self.dataset_path)

    def run(self):
        for data_id in range(self.data_num):
            print(f"To generate dataset {data_id}...")

            # Step 1: Define Gaussian distribution parameters
            x_mean, x_std = 6000, 2000  # Centered at 6000, std of 2000
            y_mean, y_std = 5000, 1600  # Centered at 5000, std of 1500
            # Set the mean and standard deviation for temporal distribution
            time_mean, time_std = 12, 4  # Centered at 12 hours, std of 4 hours

            # Step 2: Generate Gaussian distributed points
            x_coords = np.clip(np.random.normal(x_mean, x_std, self.num_points), 0, 12000)  # meter
            y_coords = np.clip(np.random.normal(y_mean, y_std, self.num_points), 0, 10000)  # meter
            time_stamps = np.clip(np.random.normal(time_mean, time_std, self.num_points), 0, 24)  # hour
            charging_demands = np.random.choice([1, 1.5, 2, 2.5, 3], self.num_points)  # hour

            demands_df = pd.DataFrame({
                'x': x_coords,
                'y': y_coords,
                'time': time_stamps,
                'demand': charging_demands,
                'step_id': 0,
            })

            # Step 3: store the plan info for each demand for each dataset
            parent_path = self.dataset_path + f'demands_dist{data_id}/'
            os.makedirs(parent_path)
            time_len_min = 24 / self.steps
            demands_df['step_id'] = (demands_df['time'] // time_len_min).astype(int)
            # Find the grids and neighbouring grids for each demand
            num_rows, num_cols = np.sqrt(self.grids), np.sqrt(self.grids)
            x_step, y_step = 12000 / num_cols, 10000 / num_rows
            demands_df['grid_x'] = (demands_df['x'] // x_step).astype(int)
            demands_df['grid_y'] = (demands_df['y'] // y_step).astype(int)
            demands_df['grid_id'] = demands_df['grid_x'] + demands_df['grid_y'] * num_cols
            demands_df = demands_df[demands_df['grid_id'] < self.grids]

            self.generate_plan_data(parent_path, demands_df)

    def generate_plan_data(self, parent_path, dataframe):

        for j in range(self.steps):
            time_path = parent_path + f'time_step{j}/'
            os.makedirs(time_path)
            # find charging demands in each time period, and add empty if no demands
            demands_df = dataframe[dataframe['step_id'] == j][['x', 'y', 'demand', 'grid_id']].reset_index(
                drop=True)
            # add plan info of neighouring stations for each demand
            for ev_agent_id, demand_info in demands_df.iterrows():
                path = time_path + f'ev_agent{ev_agent_id}.csv'
                # find charging demands within the current time period
                grid_id = int(demand_info['grid_id'])  # grid id, or MARL agent id
                stations_info_df = self.world.grid_with_neighbors[grid_id].copy()  # stations info for the grid

                # generate plans for each charging demand
                for plan_id, station_info in stations_info_df.iterrows():
                    # only plan the demand to available charging slot
                    slot_dim = int(station_info['dim_id'])
                    # calculate relative distance between demand loc and station
                    relative_distance = np.sqrt((demand_info['x'] - station_info['x']) ** 2 +
                                                ((demand_info['y'] - station_info['y']) ** 2))
                    plan_cost = relative_distance / (10000 / 5 * 2 * 1.4)
                    stations_info_df.loc[plan_id, 'cost'] = plan_cost

                output_df = stations_info_df.copy()
                output_df['demand'] = demand_info['demand']
                output_df.to_csv(path, index=False)
