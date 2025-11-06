import os
import random
import subprocess
import shutil
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from env.RealWorld import RealWorld


class Benchmark:

    def __init__(self, env_config):
        # Number of repetitions / episodes
        self.repetition = env_config["repetition"]
        # Number of time periods in an episode
        self.steps = env_config["steps"]
        # Input behaviour value to balance discomfort and inefficiency
        self.beta = env_config["beta"]
        # The probability of charging slots availability
        self.avail_prob = env_config["slots_avail"]
        # Whether EVs to observe the current queueing time of charging stations
        self.is_observed = env_config["IsObserved"]
        # Whether to recommend the behavior of EVs over time
        self.is_recommend = env_config["IsRecommended"]

        # INITIALIZE THE ENV
        self.world = RealWorld(avail_prob=self.avail_prob)
        # Plan size
        self.dimension = len(self.world.station_data)
        # Time length of each time period
        self.time_len = 24 / self.steps
        # EVs demands info per time period
        self.demands_per_time_arr = None
        # Remained queueing/charging time of each charging station
        self.remained_charging_time = None
        # Remained queueing/charging time of each charging slot
        self.remained_time_slots = None
        # Accumulated demanded charging time of each charging station
        self.demands_time_total = None

    def run(self):
        metric_names = ['discomfort', 'inefficiency', 'overall', 'unfairness',
                        'traveling_distance', 'queuing_time', 'waiting_time']
        output_data_list = {name: [] for name in metric_names}
        demands_list, max_energy_demands, cost_time_list = [], [], []
        parent_path = os.path.join(os.getcwd(), f'output/Avail{self.avail_prob}_Steps{self.steps}_Beta{self.beta}_AP{misbehaved_ratio}/')
        if os.path.exists(parent_path):
            shutil.rmtree(parent_path)
        os.mkdir(parent_path)

        # Construct the Model to predict the number of EVs in the next period
        data_for_train = []
        for data_id in range(len(self.world.data_train)):
            demands_df = self.world.data_train[data_id]
            demands_df['step_id'] = (demands_df['time'] // self.time_len).astype(int)
            hour_counts_series = demands_df['step_id'].value_counts().sort_index()
            full_range = pd.Series(0, index=range(0, self.steps+1))
            hour_counts_list = full_range.add(hour_counts_series, fill_value=0).astype(int).to_list()
            data_for_train.append(hour_counts_list)
        models = linear_prediction_models(data_for_train)

        # Start to run the process
        for iter_id in range(self.repetition):
            demands_df = self.world.data_train[iter_id]

            slots_num_arr = self.world.station_data['slot_num'].to_numpy().astype(int)
            self.remained_charging_time = np.zeros(self.dimension, dtype=np.float32)
            self.demands_time_total = np.zeros(self.dimension, dtype=np.float32)
            self.remained_time_slots = np.zeros(shape=(self.dimension, max(slots_num_arr)), dtype=np.float32)
            for i, count in enumerate(slots_num_arr):
                self.remained_time_slots[i, count:] = 48

            # Randomly set the misbehaved/selfish agents
            demands_df['misbehaved'] = np.zeros(len(demands_df))
            num_to_set = int(len(demands_df) * misbehaved_ratio)
            sample_indices = demands_df.sample(n=num_to_set, random_state=42).index
            demands_df.loc[sample_indices, 'misbehaved'] = 1

            # Assign demands to different time periods
            demands_df['step_id'] = (demands_df['time'] // self.time_len).astype(int)
            self.demands_per_time_arr = []
            for i in range(self.steps):
                # find charging demands in each time period, and add empty if no demands
                dataframe = demands_df[demands_df['step_id'] == i][['x', 'y', 'time', 'demand', 'misbehaved']]
                dataframe['time'] -= i * self.time_len
                dataframe = dataframe.sort_values(by='time').reset_index(drop=True)
                self.demands_per_time_arr.append(dataframe)

            # run the for each step / time period
            avg_discomfort_cost, inefficiency_cost, unfairness = [], [], []
            traveling_distance_total, queueing_time_total = 0, 0
            for step_id in range(self.steps):

                # Execute The STEP
                metrics, state = self.step(step_id, models[step_id], data_for_train[iter_id])

                print(f"For repetition {iter_id} step {step_id}, "
                      f"discomfort: {round(metrics[0], 3)}, inefficiency: {round(metrics[1], 3)}, "
                      f"traveling: {round(metrics[2], 3)}, queueing: {round(metrics[3], 3)}, "
                      f"unfairness: {round(metrics[4], 3)}")
                avg_discomfort_cost.append(metrics[0])
                inefficiency_cost.append(metrics[1])
                traveling_distance_total += metrics[2]
                queueing_time_total += metrics[3]
                unfairness.append(metrics[4])

            output_data_list['discomfort'].append(sum(avg_discomfort_cost) / self.steps)
            output_data_list['inefficiency'].append(sum(inefficiency_cost) / self.steps)
            overall_cost = (sum(avg_discomfort_cost) + sum(inefficiency_cost)) / self.steps
            output_data_list['overall'].append(overall_cost)
            output_data_list['unfairness'].append(sum(unfairness) / self.steps)
            output_data_list['traveling_distance'].append(traveling_distance_total / len(demands_df) / 1000)
            output_data_list['queuing_time'].append(queueing_time_total / len(demands_df))
            waiting_time_mean = (traveling_distance_total / 30 / 1000 + queueing_time_total) / len(demands_df)
            output_data_list['waiting_time'].append(waiting_time_mean)
            demands_list.append(state)
            max_energy_demands.append(max(state) * 3.6 * 1.8)
            cost_time_list.append(avg_discomfort_cost + inefficiency_cost)

        # Output to csv file
        output_df = pd.DataFrame()
        output_stores = []
        for name in metric_names:
            output_df[name] = np.array(output_data_list[name])
            output_stores.append(np.mean(output_data_list[name]))
        for name in metric_names:
            output_df.loc['AVG', name] = np.mean(output_data_list[name])
            output_df.loc['STD', name] = np.std(output_data_list[name])
        output_df.to_csv(parent_path + 'Metrics.csv')

        output_state_df = pd.DataFrame(demands_list)
        output_state_df.to_csv(parent_path + 'Demands.csv', index=False, header=False)

        output_cost_df = pd.DataFrame(cost_time_list)
        output_cost_df.to_csv(parent_path + 'Cost_time.csv', index=False, header=False)

        output_stores.append(np.mean(max_energy_demands))
        return output_stores

    def step(self, step_id, prediction_model, numAgents_list):
        avg_discomfort_cost = 0
        inefficiency_cost = 0
        charging_demands_df = self.demands_per_time_arr[step_id]
        parent_path = os.path.join(os.getcwd(), 'datasets/EVdemands/')
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

                # Step 3: store information
                agent_path = parent_path + f'agent_{ev_agent_id}.plans'
                with open(agent_path, 'w', newline='', encoding='utf-8') as outFile:
                    outFile.write(plans_string)
                outFile.close()
                selected_plans[ev_agent_id] = stations_info_df.to_dict(orient='records')

            # Step 4: run EPOS and obtain results, if agent > 1
            if numAgents > 1:
                # SET BEHAVIOR
                if self.is_recommend:
                    if step_id < 1:
                        past_agentsNum = [numAgents, numAgents, numAgents]
                    elif step_id < 2:
                        past_agentsNum = [numAgents, numAgents, numAgents_list[step_id - 1]]
                    else:
                        past_agentsNum = [numAgents, numAgents_list[step_id - 1], numAgents_list[step_id - 2]]
                    future_agentsNum = prediction_model.predict([past_agentsNum])[0]
                    avail_slots_num = np.sum(self.remained_time_slots == 0)
                    behavior = avail_slots_num / (numAgents + future_agentsNum) / 8
                    behaviors = np.ones(numAgents) * min(round(behavior, 2), 1)

                    # ! added
                    misbehaved_arr = charging_demands_df['misbehaved'].to_numpy()
                    behaviors[misbehaved_arr == 1] = 1

                else:
                    behaviors = np.ones(numAgents) * self.beta

                path_behavior = os.path.join(os.getcwd(), f'datasets/EVdemands/behaviours.csv')
                behavior_df = pd.DataFrame(columns=['idx', 'alpha', 'beta'])
                behavior_df['idx'] = np.array(range(len(behaviors)))
                behavior_df['alpha'] = np.zeros(len(behaviors), dtype=np.int32)
                behavior_df['beta'] = behaviors
                behavior_df.to_csv(path_behavior, index=False, header=False)

                command = ['java', '-jar', 'IEPOS_input.jar', f'{numAgents}']
                epos_output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
                epos_output = epos_output.replace('\n', '').split(',')
                while not epos_output[0].isdigit():
                    epos_output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
                    epos_output = epos_output.replace('\n', '').split(',')
                selected_plan_indexes = np.array(
                    [int(x) for x in epos_output[-self.dimension - 2 - numAgents:-self.dimension - 2]])
                local_cost = float(epos_output[-self.dimension - 2])
                global_cost = float(epos_output[-self.dimension - 1])
                response = np.array([float(x) for x in epos_output[-self.dimension:]])
            else:
                # if only one agent, then randomly find the plan that charge at an available slot
                selected_idx_random = random.randint(0, len(selected_plans[0]) - 1)
                selected_plan_indexes = [selected_idx_random]
                local_cost = selected_plans[0][selected_idx_random]['cost']
                response = selected_plans[0][selected_idx_random]['plan']

            # Step 5: Calculate the avg discomfort cost and inefficiency cost
            avg_discomfort_cost = local_cost
            inefficiency_cost = np.sqrt(np.mean(response ** 2))
            # inefficiency_cost = np.var(response)

        # Step 6: update the state of charging stations
        traveling_distance_total = 0
        queueing_time_total = 0
        local_cost_arr = []
        for ev_agent_id, demand_info in charging_demands_df.iterrows():
            request_time = demand_info['time']
            occupied_time = demand_info['demand']
            # find the selected plan that contains the dict info of the charging statiopn
            selected_plan_dict = selected_plans[ev_agent_id][selected_plan_indexes[ev_agent_id]]
            # store the local cost of all agents to calculate unfairness of discomfort
            local_cost_arr.append(selected_plan_dict['cost'])
            # update traveling distance
            traveling_distance_total += selected_plan_dict['travel']
            # find the charging slot of the selected station with the minimum queueing time
            selected_station_id = int(selected_plan_dict['dim_id'])
            slots_in_station = self.remained_time_slots[selected_station_id]
            min_indices = np.where(slots_in_station == np.min(slots_in_station))[0]
            # update total queueing time
            queueing_time_total += max(slots_in_station[min_indices[0]] - request_time, 0)
            # update queueing time of charging slots
            slots_in_station[min_indices[0]] += request_time + occupied_time
            # obtain the new queueing time for stations
            min_indices_new = np.where(slots_in_station == np.min(slots_in_station))[0]
            self.remained_charging_time[selected_station_id] = slots_in_station[min_indices_new[0]]
            # update the output of total demanding on charging stations
            self.demands_time_total[selected_station_id] += occupied_time

        # Minus passing time
        slots_num_arr = self.world.station_data['slot_num'].to_numpy()
        self.remained_charging_time -= np.ones(self.dimension) * slots_num_arr * self.time_len
        self.remained_charging_time[self.remained_charging_time < 0] = 0
        self.remained_time_slots -= np.ones_like(self.remained_time_slots) * self.time_len
        self.remained_time_slots[self.remained_time_slots < 0] = 0

        unfairness = gini_coefficient(local_cost_arr) if sum(local_cost_arr) > 0 else 0
        metrics = [avg_discomfort_cost, inefficiency_cost, traveling_distance_total, queueing_time_total, unfairness]
        state = self.demands_time_total.tolist()
        return metrics, state


def linear_prediction_models(data_vectors):
    data_vectors = np.array(data_vectors)
    models_num = len(data_vectors[0]) - 1
    models = []

    for m_id in range(models_num):

        X = []  # input windows
        y = []  # next values (targets)

        for data_id in range(data_vectors.shape[0]):

            if m_id < 1:
                X.append([data_vectors[data_id][m_id], data_vectors[data_id][m_id], data_vectors[data_id][m_id]])
            elif m_id < 2:
                X.append([data_vectors[data_id][m_id], data_vectors[data_id][m_id], data_vectors[data_id][m_id - 1]])
            else:
                X.append(
                    [data_vectors[data_id][m_id], data_vectors[data_id][m_id - 1], data_vectors[data_id][m_id - 2]])
            y.append(data_vectors[data_id][m_id + 1])

        model = LinearRegression()
        model.fit(np.array(X), np.array(y))
        models.append(model)

    return models


def gini_coefficient(values):
    values = np.array(values)
    sorted_values = np.sort(values)
    n = len(values)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_values))
            - (n + 1) * np.sum(sorted_values)) / (n * np.sum(sorted_values))
    return gini


def standard_unfairness(values):
    values = np.array(values)
    return np.std(values) / np.mean(values)


if __name__ == '__main__':

    config = {
        "repetition": 100,      # Set the number of repetitions
        "steps": 12,            # Set the number of time windows
        "slots_avail": 0.5,     # Set the ratio of available charging slots over total number of slots
        "beta": 0.5,            # Set the behavior value
        "IsObserved": True,     # Whether the discomfort includes the observed queuing time
        "IsRecommended": True,  # Whether to use behavior recommendation mechanism
    }
    misbehaved_ratio = 0  # Set the fraction of adversarial EVs

    bench = Benchmark(config)
    outputs = bench.run()


