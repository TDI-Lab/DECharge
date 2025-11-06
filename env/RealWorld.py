import pandas as pd
import numpy as np
import os


class RealWorld:

    def __init__(self, avail_prob=1):
        # 1. Method to distribute spatio-temporal EVs: Read dataset
        data_path_parent = os.path.join(os.getcwd(), f'datasets/ChargingDemands/')
        data_generated = []
        for file in os.listdir(data_path_parent):
            if '_06_2022' in file or '_07_2022' in file or '_08_2022' in file or '_09_2022' in file \
                    or '_05_2022' in file or '_04_2022' in file or '_03_2022' in file:
                path = data_path_parent + file
                df = pd.read_csv(path)
                data_generated.append(df)
        self.data_train = data_generated
        self.data_test = data_generated[int(len(data_generated) * 0.7):]

        # 2. Set features to station, including x, y
        self.station_data = pd.read_csv(os.getcwd() + f'/datasets/Stations/info_static_{avail_prob}.csv')
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

        # 2.3. Add dimension index
        stations_num = len(self.station_data)
        self.station_data = self.station_data.sort_values(by=['s_id'])
        self.station_data['grid_id'] = range(stations_num)
        self.station_data['dim_id'] = range(stations_num)

        # 2.4. Add the cost column and plan column
        self.station_data['cost'] = np.zeros(stations_num)
        self.station_data['travel'] = np.zeros(stations_num)
        plans = np.eye(stations_num)  # create an identity matrix where each row is a one-hot vector
        self.station_data['plan'] = list(plans)
        self.station_data = self.station_data.reset_index()

        # # 3. Find the neighbours
        # self.grid_with_neighbors = []
        # search_range = neighbor_num * 3600
        # important_features = ['dim_id', 'grid_id', 'x', 'y', 'slot_num', 'travel', 'cost', 'plan']
        # self.station_data['dist_to_center'] = np.zeros(stations_num)
        # for station_id in range(stations_num):
        #     center_x = self.station_data.loc[self.station_data['grid_id'] == station_id, 'x'].values[0]
        #     center_y = self.station_data.loc[self.station_data['grid_id'] == station_id, 'y'].values[0]
        #     self.station_data['dist_to_center'] = np.sqrt((self.station_data['x'] - center_x) ** 2 +
        #                                                   (self.station_data['y'] - center_y) ** 2)
        #     points_df = self.station_data[self.station_data['dist_to_center'] < search_range][important_features].copy()
        #     points_df[['dim_id', 'grid_id']] = points_df[['dim_id', 'grid_id']].astype(int)
        #     points_df = points_df.reset_index(drop=True)
        #     self.grid_with_neighbors.append(points_df)
