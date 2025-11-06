import os
import random
import subprocess
import shutil
import numpy as np
import pandas as pd

import gym
from gym import spaces
from env.ParisRealWorld import ParisRealWorld


class EVRealEnv(gym.Env):

    def __init__(self, env_config):
        # INPUT
        # Number of action dims
        self.a_dim = env_config["a_dim"]
        # Number of steps in an episode
        self.steps = env_config["steps"]
        # The probability of charging slots availability
        avail_prob = env_config["slots_avail"]
        # Whether to use EPOS to coordinate electric vehicles
        self.is_observed = env_config["IsObserved"]
        # Tradeoff weight between global and local cost
        self.sigma = env_config["sigma"]
        # Number of the agents
        self.n = 1

        # INITIALIZE THE ENV
        self.time_len = 24 // self.steps
        self.world = ParisRealWorld(avail_prob=avail_prob)
        self.slots_availability = None
        # Plan dimension = # of charging stations
        self.dimension = len(self.world.station_data)
        # Charging demands in each time period
        self.demands_per_time_arr = None
        # The remained charging time of charging stations
        self.remained_charging_time = None
        # The list of remained charging time of charging slots
        self.remained_time_slots = None
        # The accumulated demand time on each charging station
        self.demands_time_total = None
        # The current step
        self.step_id = None

        # Generate fundamental plans for EVs
        # self.generate_plans()
        # # Generate dataset for each demand
        # data_generation = DatasetGeneration(grids=self.n, steps=self.steps,
        #                                     data_num=40, num_points=600, world=self.world)
        # data_generation.run()

        # ACTION AND OBSERVATION
        self.action_space = []
        self.observation_space = []
        for a in range(self.n):
            # Dimension space of action (D x D matrix) and observation (D vector)
            self.action_space.append(spaces.Discrete(self.a_dim))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf,
                                                     shape=(5,), dtype=np.float32))

    def reset(self, is_test=False):
        slots_num_arr = self.world.station_data['slot_num'].to_numpy().astype(int)
        self.remained_charging_time = np.zeros(self.dimension, dtype=np.float32)
        self.demands_time_total = np.zeros(self.dimension, dtype=np.float32)
        # set the remained occupied time for each slot, set infinite/high value to down slots
        self.remained_time_slots = np.zeros(shape=(self.dimension, max(slots_num_arr)), dtype=np.float32)
        self.slots_availability = np.ones(self.dimension, dtype=np.int32) * slots_num_arr
        for i, count in enumerate(slots_num_arr):
            self.remained_time_slots[i, count:] = 48

        # Read the charging demands dataset
        if is_test:
            demands_df = random.choice(self.world.data_test).copy()
        else:
            demands_df = random.choice(self.world.data_train).copy()

        # Assign demands to different time periods
        demands_df['step_id'] = (demands_df['time'] // self.time_len).astype(int)
        self.demands_per_time_arr = []
        for i in range(self.steps):
            # find charging demands in each time period, and add empty if no demands
            dataframe = demands_df[demands_df['step_id'] == i][['x', 'y', 'time', 'demand']]
            dataframe['time'] -= i * self.time_len
            dataframe = dataframe.sort_values(by='time').reset_index(drop=True)
            self.demands_per_time_arr.append(dataframe)

        obs_n = []
        self.step_id = 0
        for agent_id in range(self.n):
            observation = [0, 0, 0, 0, 0]
            obs_n.append(observation)

        return obs_n

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        charging_demands_df = self.demands_per_time_arr[self.step_id]
        parent_path = os.path.join(os.getcwd(), 'datasets/EVdemands/')
        # selected group of plans for each EV agent, each group is a list of plan dict [group_id, dim_id, cost, plan]
        selected_plans = {}

        if not charging_demands_df.empty:
            numAgents = len(charging_demands_df)

            for ev_agent_id, demand_info in charging_demands_df.iterrows():

                # Step 1: find charging demands within the current time period
                stations_info_df = self.world.station_data
                stations_info_df = stations_info_df[stations_info_df['slot_num'] > 0].copy()

                # Step 2: generate plans for each charging demand
                plans_string = ''
                for plan_id, station_info in stations_info_df.iterrows():
                    relative_distance = np.sqrt((demand_info['x'] - station_info['x']) ** 2 +
                                                ((demand_info['y'] - station_info['y']) ** 2))
                    traveling_distance = relative_distance
                    stations_info_df.loc[plan_id, 'travel'] = traveling_distance
                    if self.is_observed:
                        slot_dim = int(station_info['dim_id'])
                        slots_in_station = self.remained_time_slots[slot_dim]
                        min_indices = np.where(slots_in_station == np.min(slots_in_station))[0]
                        queueing_time = max(slots_in_station[min_indices[0]] - demand_info['time'], 0)
                        plan_cost = traveling_distance / 5000 + queueing_time
                    else:
                        plan_cost = traveling_distance / 1000
                    stations_info_df.loc[plan_id, 'cost'] = plan_cost

                    plan = np.zeros(self.dimension) + self.remained_charging_time / numAgents
                    plan[int(station_info['dim_id'])] += demand_info['demand']
                    plan_str = str(plan_cost) + ':' + ','.join(map(str, plan))
                    plans_string = plans_string + plan_str + '\n'

                # Step 3: group the plans and only use group of plans according to the action
                agent_path = parent_path + f'agent_{ev_agent_id}.plans'
                with open(agent_path, 'w', newline='', encoding='utf-8') as outFile:
                    outFile.write(plans_string)
                outFile.close()
                selected_plans[ev_agent_id] = stations_info_df.to_dict(orient='records')

            # Step 4: run EPOS and obtain results, if agent > 1
            if numAgents > 1:
                behaviors = np.ones(numAgents) * self.action_to_beta(action_n[0])
                path_behavior = os.path.join(os.getcwd(), f'datasets/EVdemands/behaviours.csv')
                behavior_df = pd.DataFrame(columns=['idx', 'alpha', 'beta'])
                behavior_df['idx'] = np.array(range(len(behaviors)))
                behavior_df['alpha'] = np.zeros(len(behaviors), dtype=np.int32)
                behavior_df['beta'] = behaviors
                behavior_df.to_csv(path_behavior, index=False, header=False)

                command = ['java', '-jar', 'IEPOS_input2.jar', f'{numAgents}']
                epos_output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
                epos_output = epos_output.replace('\n', '').split(',')
                while not epos_output[0].isdigit():
                    epos_output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
                    epos_output = epos_output.replace('\n', '').split(',')
                selected_plan_indexes = np.array(
                    [int(x) for x in epos_output[-self.dimension - 2 - numAgents:-self.dimension - 2]])
                local_cost = float(epos_output[-self.dimension - 2])
                response = np.array([float(x) for x in epos_output[-self.dimension:]])
            else:
                #  randomly find the plan that charge at an available station
                selected_plan_indexes = []
                local_cost = 0
                response = np.zeros(self.dimension)
                for ev_agent_id in range(numAgents):
                    selected_idx_random = random.randint(0, len(selected_plans[ev_agent_id]) - 1)
                    selected_plan_indexes.append(selected_idx_random)
                    local_cost += selected_plans[ev_agent_id][selected_idx_random]['cost']
                    response = response + selected_plans[ev_agent_id][selected_idx_random]['plan']
                local_cost = local_cost / numAgents

            # Calculate the mean waiting time of EVs (avg discomfort cost)
            avg_discomfort_cost = local_cost
            inefficiency_cost = np.sqrt(np.mean(response ** 2))

        else:
            avg_discomfort_cost = 0
            inefficiency_cost = 0

        # Step 6: update the state of charging stations (occupied time = travel time + charging time)
        # increase the new occupied time
        traveling_distance_total = 0
        queueing_time_total = 0
        for ev_agent_id, demand_info in charging_demands_df.iterrows():
            request_time = demand_info['time']
            occupied_time = demand_info['demand']
            selected_plan_dict = selected_plans[ev_agent_id][selected_plan_indexes[ev_agent_id]]
            traveling_distance_total += selected_plan_dict['travel']
            selected_station_id = int(selected_plan_dict['dim_id'])
            slots_in_station = self.remained_time_slots[selected_station_id]
            min_indices = np.where(slots_in_station == np.min(slots_in_station))[0]
            queueing_time_total += max(slots_in_station[min_indices[0]] - request_time, 0)
            slots_in_station[min_indices[0]] += request_time + occupied_time
            min_indices_new = np.where(slots_in_station == np.min(slots_in_station))[0]
            self.remained_charging_time[selected_station_id] = slots_in_station[min_indices_new[0]]
            self.demands_time_total[selected_station_id] += occupied_time
        traveling_distance_total = traveling_distance_total / 1000 / numAgents if numAgents > 0 else 0
        queueing_time_total = queueing_time_total / numAgents if numAgents > 0 else 0
        # reduce the current time
        slots_num_arr = self.world.station_data['slot_num'].to_numpy()
        self.remained_charging_time -= np.ones(self.dimension) * slots_num_arr * 24 / self.steps
        self.remained_charging_time[self.remained_charging_time < 0] = 0
        self.remained_time_slots -= np.ones_like(self.remained_time_slots) * 24 / self.steps
        self.remained_time_slots[self.remained_time_slots < 0] = 0
        # Update the availability of charging slots by summing the number of unoccupied slots
        for station_id in range(self.dimension):
            self.slots_availability[station_id] = np.sum(self.remained_time_slots[station_id] == 0)

        # Output Results
        self.step_id += 1
        for agent_id in range(self.n):
            # Update Observation, observe the demands and stations in the current grid (if it has)
            observation = [self.step_id, avg_discomfort_cost, inefficiency_cost, traveling_distance_total,
                           queueing_time_total]
            obs_n.append(observation)

            # Calculate Reward
            # reward = - self.sigma * traveling_time_total - (1 - self.sigma) * queueing_time_total
            reward = self.reward_calc(avg_discomfort_cost, inefficiency_cost)
            reward_n.append(reward)

            done_n.append(False)

        print(f"Step: {self.step_id}; Reward: {reward}")
        state_info = self.demands_time_total.tolist()
        action_info = action_n.tolist() + [avg_discomfort_cost, inefficiency_cost]
        info_n.append(np.array([avg_discomfort_cost, inefficiency_cost, traveling_distance_total, queueing_time_total]))
        info_n.append(state_info)
        info_n.append(action_info)
        return obs_n, reward_n, done_n, info_n

    def generate_plans(self):
        data_path_parent = os.path.join(os.getcwd(), f'datasets/EVdemands/')
        shutil.rmtree(data_path_parent)  # Deletes the directory and everything inside
        os.makedirs(data_path_parent)  # Recreates an empty directory
        for agent_id in range(self.n):
            plan_path = data_path_parent + f'agent_{agent_id}.plans'
            grid_id = agent_id
            stations_info_df = self.world.grid_with_neighbors[grid_id]

            # generate all possible plans for each grid/MARL agent
            plans_arr = []
            plans_to_write = ''
            for plan_id, station_info in stations_info_df.iterrows():
                plan = np.zeros(self.dimension, dtype=np.int32)
                plan[int(station_info['dim_id'])] = 1
                plans_arr.append(plan)

                plan_format = '0.0:' + ','.join(map(str, plan)) + '\n'
                plans_to_write = plans_to_write + plan_format
                with open(plan_path, 'w', newline='', encoding='utf-8') as outFile:
                    outFile.write(plans_to_write)
                outFile.close()

    def get_observation(self, grid_id, agent_idx_for_evs_next):
        stations_in_neighbours = self.world.grid_with_neighbors[grid_id]
        stations_in_grid = stations_in_neighbours[stations_in_neighbours['grid_id'] == grid_id]
        neighbouring_grids = np.unique(stations_in_neighbours['grid_id'].to_numpy())
        # number of charging demands within the current grid
        demands_num_in_grid = agent_idx_for_evs_next.count(grid_id)
        # number of charging demands within neighbouring grids
        demands_num_neighbours = sum([agent_idx_for_evs_next.count(idx) for idx in neighbouring_grids])
        # number of available charging slots within the current grid
        available_slots_in_grid = np.sum(self.slots_availability[stations_in_grid['dim_id'].to_list()])
        # number of available charging slots within neighbouring grids
        available_slots_neighbours = np.sum(self.slots_availability[stations_in_neighbours['dim_id'].to_list()])
        # total remained charging time within the current grid
        total_remained_time_in_grid = np.sum(self.remained_charging_time[stations_in_grid['dim_id'].to_list()])
        # total remained charging time within neighbouring grids
        total_remained_time_neighbours = np.sum(self.remained_charging_time[stations_in_neighbours['dim_id'].to_list()])

        observation = [demands_num_in_grid, demands_num_neighbours, available_slots_in_grid, available_slots_neighbours,
                       total_remained_time_in_grid, total_remained_time_neighbours]
        return observation

    def action_to_beta(self, action):
        beta_disparity = 1 / self.a_dim

        return 0.1 + round(action * beta_disparity, 2)

    def reward_calc(self, discomfort, inefficiency):
        # sigmoid_discomfort = 1 / (1 + np.exp(-discomfort))
        # sigmoid_inefficiency = 1 / (1 + np.exp(-inefficiency))

        # reward = - self.sigma * sigmoid_discomfort - (1 - self.sigma) * sigmoid_inefficiency
        reward = - self.sigma * discomfort - (1 - self.sigma) * inefficiency
        return reward
