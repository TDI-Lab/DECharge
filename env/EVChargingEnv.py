import math
import os
import random
import subprocess
import shutil
import numpy as np
import pandas as pd

import gym
from gym import spaces
from pathlib import Path

from env.ParisWorld import ParisWorld
from env.DatasetGeneration import DatasetGeneration


class EVChargingEnv(gym.Env):

    def __init__(self, env_config):
        # INPUT
        # Number of the agents
        self.n = env_config["agents"]
        # Number of action dims
        self.a_dim = env_config["a_dim"]
        # Number of steps in an episode
        self.steps = env_config["steps"]
        # Number of electric vehicles passing per hour
        self.vehicles = env_config["vehicles"]
        # The time type (weekend or weekday) of charging demands distribution
        time_style = env_config["time_style"]
        # The probability of charging slots availability
        avail_prob = env_config["slots_avail"]
        # The number of neighbouring grids as the radius for station searching of each EV
        neighbours_search = env_config["neighbours_search"]
        # The style to divide grids: square, voronoi
        self.grid_style = env_config["grid_style"]
        # Whether to use EPOS to coordinate electric vehicles
        self.is_epos = env_config["is_epos"]
        # Tradeoff weight between global and local cost
        self.sigma = env_config["sigma"]

        # INITIALIZE THE ENV
        self.world = ParisWorld(grids=self.n, vehicles=self.vehicles, data_read_num=40,
                                time_style=time_style, grid_style=self.grid_style,
                                avail_prob=avail_prob, neighbor_num=neighbours_search)
        # Plan dimension = # of charging stations
        self.dimension = len(self.world.station_data)
        # Charging demands in each time period
        self.demands_per_time_arr = None
        # The system target to calculate inefficiency cost
        self.target = None
        # The availability of charging slots, 1 is available 0 is not
        self.slots_availability = None
        # The remained charging time of charging stations
        self.remained_charging_time = None
        # The list of remained charging time of charging slots
        self.remained_time_slots = None
        # The accumulated demand time on each charging station
        self.demands_time_total = None
        # The current step
        self.step_id = None

        # Generate fundamental plans for EVs
        self.generate_plans()
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
                                                     shape=(9,), dtype=np.float32))

    def reset(self, is_test=False):
        slots_num_arr = self.world.station_data['slot_num'].to_numpy()
        self.slots_availability = np.ones(self.dimension, dtype=np.int32) * slots_num_arr
        self.target = self.slots_availability.copy()
        self.remained_charging_time = np.zeros(self.dimension, dtype=np.float32)
        self.demands_time_total = np.zeros(self.dimension, dtype=np.float32)
        # set the remained occupied time for each slot, set infinite/high value to down slots
        self.remained_time_slots = np.zeros(shape=(self.dimension, max(slots_num_arr)), dtype=np.float32)
        for i, count in enumerate(slots_num_arr):
            self.remained_time_slots[i, count:] = 48

        # Read the charging demands dataset
        if is_test:
            demands_df = random.choice(self.world.data_test).copy()
        else:
            demands_df = random.choice(self.world.data_train).copy()

        # Find the grids and neighbouring grids for each demand
        input_data = demands_df[['x', 'y']]
        if self.grid_style == 'square':
            demands_df['grid_id'] = self.world.find_square_index(input_data)
        elif self.grid_style == 'voronoi':
            demands_df['grid_id'] = self.world.find_voronoi_index(input_data)
        else:
            print("Wrong grid type!!!")
            return
        demands_df = demands_df[demands_df['grid_id'] < self.n]

        # Assign demands to different time periods
        time_len_min = 24 // self.steps
        demands_df['step_id'] = (demands_df['time'] // time_len_min).astype(int)
        self.demands_per_time_arr = []
        for i in range(self.steps):
            # find charging demands in each time period, and add empty if no demands
            dataframe = demands_df[demands_df['step_id'] == i][['x', 'y', 'demand', 'grid_id']].reset_index(drop=True)
            self.demands_per_time_arr.append(dataframe)

        obs_n = []
        self.step_id = 0
        agent_idx_for_evs_next = self.demands_per_time_arr[self.step_id]['grid_id'].to_list()
        for agent_id in range(self.n):
            observation = [agent_id, self.step_id, 0] + self.get_observation(agent_id, agent_idx_for_evs_next)
            obs_n.append(observation)

        return obs_n

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        charging_demands_df = self.demands_per_time_arr[self.step_id]
        demand_time_dict = {}  # list of demand time
        agent_indexes_for_evs = []  # list of indexes of MARL agents/grids for each EV agent
        plan_indexes_input = ''  # selected indexes of plans for EPOS selection
        costs_input = ''  # selected costs of plans for EPOS selection
        # selected group of plans for each EV agent, each group is a list of plan dict [group_id, dim_id, cost, plan]
        selected_plans = {}

        if not charging_demands_df.empty:

            behaviors = []
            for ev_agent_id, demand_info in charging_demands_df.iterrows():

                # Step 1: find charging demands within the current time period
                grid_id = int(demand_info['grid_id'])  # grid id, or MARL agent id
                stations_info_df = self.world.grid_with_neighbors[grid_id].copy()  # stations info for the grid
                action = int(action_n[grid_id])  # the action of the MARL agent, or selected plans group id
                behaviors.append(action)
                demand_time_dict[ev_agent_id] = demand_info['demand']  # charging time for this demand
                agent_indexes_for_evs.append(grid_id)

                # Step 2: generate plans for each charging demand
                for plan_id, station_info in stations_info_df.iterrows():
                    # only plan the demand to available charging slot
                    slot_dim = int(station_info['dim_id'])
                    # if self.slots_availability[slot_dim] > 0:
                    # calculate traveling time between demand loc and station
                    relative_distance = np.sqrt((demand_info['x'] - station_info['x']) ** 2 +
                                                ((demand_info['y'] - station_info['y']) ** 2))
                    traveling_time = relative_distance / 1000 / 4
                    stations_info_df.loc[plan_id, 'travel'] = traveling_time
                    # calculate the queueing time on the selected station
                    slots_in_station = self.remained_time_slots[slot_dim]
                    min_indices = np.where(slots_in_station == np.min(slots_in_station))[0]
                    queueing_time = slots_in_station[min_indices[0]]
                    # calculate the waiting time as the cost of the plan
                    plan_cost = traveling_time + queueing_time
                    stations_info_df.loc[plan_id, 'cost'] = plan_cost

                # Step 3: group the plans and only use group of plans according to the action
                if self.is_epos:
                    selected_generated_plans = stations_info_df[stations_info_df['cost'] > 0].copy()
                else:
                    generated_plans_df = stations_info_df[stations_info_df['cost'] > 0].copy()
                    if len(generated_plans_df) < self.a_dim:
                        generated_plans_df = stations_info_df.copy()
                    # divide on average if plan num is not smaller than action num
                    generated_plans_df = generated_plans_df.sort_values(by='cost')
                    # generated_plans_df['rank'] = generated_plans_df['cost'].rank(method='average')
                    # generated_plans_df['group_id'] = pd.qcut(generated_plans_df['rank'], q=self.a_dim, labels=False)
                    group_size = len(generated_plans_df) // self.a_dim
                    generated_plans_df['rank'] = np.array(range(len(generated_plans_df)))
                    generated_plans_df['group_id'] = (generated_plans_df['rank'] // group_size)
                    selected_generated_plans = generated_plans_df[generated_plans_df['group_id'] <= action]

                # Step 4: store information
                plan_indexes_input = plan_indexes_input + ';' + ','.join(
                    [str(index) for index in selected_generated_plans.index])
                costs_input = costs_input + ';' + ','.join([str(round(c, 3)) for c in selected_generated_plans['cost']])
                selected_plans[ev_agent_id] = selected_generated_plans.to_dict(orient='records')

            # Step 5: run EPOS and obtain results, if agent > 1
            # input numAgents, plan costs, target
            numAgents = len(agent_indexes_for_evs)
            if self.is_epos and numAgents > 1:
                beta_action = np.vectorize(self.action_to_beta)(behaviors)
                path_behavior = os.path.join(os.getcwd(), f'datasets/EVdemands/behaviours.csv')
                behavior_df = pd.DataFrame(columns=['idx', 'alpha', 'beta'])
                behavior_df['idx'] = np.array(range(len(behaviors)))
                behavior_df['alpha'] = np.zeros(len(behaviors), dtype=np.int32)
                behavior_df['beta'] = beta_action
                behavior_df.to_csv(path_behavior, index=False, header=False)

                agent_indexes_input = ','.join([str(a) for a in agent_indexes_for_evs])
                target_input = ','.join([str(t) for t in self.target])
                command = ['java', '-jar', 'IEPOS_input.jar',
                           f'{agent_indexes_input}', f'{plan_indexes_input}', f'{costs_input}', f'{target_input}']
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
            # Update the target
            self.target = self.slots_availability.copy()
            self.target = np.minimum(self.target, response)  # only consider excessive demands
            inefficiency_cost = np.sum((self.target - response) ** 2)

        else:
            avg_discomfort_cost = 0
            numAgents = 0
            inefficiency_cost = 0

        # Step 6: update the state of charging stations (occupied time = travel time + charging time)
        # reduce the current time
        slots_num_arr = self.world.station_data['slot_num'].to_numpy()
        self.remained_charging_time -= np.ones(self.dimension) * slots_num_arr * 24 / self.steps
        self.remained_charging_time[self.remained_charging_time < 0] = 0
        self.remained_time_slots -= np.ones_like(self.remained_time_slots) * 24 / self.steps
        self.remained_time_slots[self.remained_time_slots < 0] = 0
        # increase the new occupied time
        traveling_time_total = 0
        queueing_time_total = 0
        for ev_agent_id in range(numAgents):
            occupied_time = demand_time_dict[ev_agent_id]
            selected_plan_dict = selected_plans[ev_agent_id][selected_plan_indexes[ev_agent_id]]
            selected_station_id = int(selected_plan_dict['dim_id'])
            slots_in_station = self.remained_time_slots[selected_station_id]
            min_indices = np.where(slots_in_station == np.min(slots_in_station))[0]
            # update the queueing time
            queueing_time_total += slots_in_station[min_indices[0]]
            # update the traveling time
            traveling_time_total += selected_plan_dict['travel']
            # update the total demanding time per charging station
            self.demands_time_total[selected_station_id] += occupied_time
            # add charging demand time to the remained occupied time for all stations
            self.remained_charging_time[selected_station_id] += occupied_time
            # add charging demand time to the remained occupied time of each slot
            slots_in_station[min_indices[0]] += occupied_time
        # Update the availability of charging slots by summing the number of unoccupied slots
        for station_id in range(self.dimension):
            self.slots_availability[station_id] = np.sum(self.remained_time_slots[station_id] == 0)

        # Calculate the load imbalance (inefficiency cost)
        # inefficiency_cost = self.inefficiency_calc()

        # Output Results
        self.step_id += 1
        if self.step_id < self.steps:
            agent_idx_for_evs_next = self.demands_per_time_arr[self.step_id]['grid_id'].to_list()
        else:
            agent_idx_for_evs_next = agent_indexes_for_evs

        for agent_id in range(self.n):
            # Update Observation, observe the demands and stations in the current grid (if it has)
            observation = [agent_id, self.step_id, avg_discomfort_cost] + self.get_observation(
                agent_id, agent_idx_for_evs_next)
            obs_n.append(observation)

            # Calculate Reward
            reward = - self.sigma * traveling_time_total - (1 - self.sigma) * queueing_time_total
            # reward = self.reward_calc(avg_discomfort_cost, inefficiency_cost)
            reward_n.append(reward)

            done_n.append(False)

        if self.is_epos:
            print(f"Step: {self.step_id}; Reward: {reward}")
        state_info = self.remained_charging_time.tolist() + self.demands_time_total.tolist()
        action_info = action_n.tolist() + [avg_discomfort_cost, inefficiency_cost]
        info_n.append(np.array([avg_discomfort_cost, inefficiency_cost, traveling_time_total, queueing_time_total]))
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

    def inefficiency_calc(self):
        function = 2
        response = self.remained_charging_time
        target = np.zeros(len(response))
        if function == 0:
            return np.var(self.remained_charging_time, ddof=1) / 10
        elif function == 1:
            y_true_scale = np.linalg.norm(target) + 1
            y_pred_scale = np.linalg.norm(response) + 1
            y_true_normalized = target / y_true_scale
            y_pred_normalized = response / y_pred_scale
            return np.sum((y_true_normalized - y_pred_normalized) ** 2)
        elif function == 2:
            return np.sqrt(np.mean((target - response) ** 2))
        elif function == 3:
            return np.sum((target - response) ** 2)
        else:
            return 0

    def action_to_beta(self, action):
        beta_disparity = 1 / self.a_dim

        return round(action * beta_disparity, 2)

    def reward_calc(self, discomfort, inefficiency):
        sigmoid_discomfort = 1 / (1 + np.exp(-discomfort))
        sigmoid_inefficiency = 1 / (1 + np.exp(-inefficiency))

        reward = - self.sigma * sigmoid_discomfort - (1 - self.sigma) * sigmoid_inefficiency
        # reward = - self.sigma * discomfort - (1 - self.sigma) * inefficiency
        return reward

