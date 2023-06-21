import numpy as np


def sb3_winterwheat_reward(output, timestep, reward_var):

    last_index_previous_state = (np.ceil(len(output) / timestep).astype('int') - 1) * timestep - 1
    wso_start = output[last_index_previous_state][reward_var]
    wso_finish = output[-1][reward_var]
    if wso_start is None: wso_start = 0.0
    if wso_finish is None: wso_finish = 0.0
    benefits = wso_finish - wso_start

    return benefits


def default_winterwheat_reward(output, env_baseline, zero_nitrogen_env_storage, timestep, reward_var):
    assert reward_var == 'TWSO' or 'WSO'

    last_index_previous_state = (np.ceil(len(output) / timestep) - 1).astype('int') * timestep - 1

    wso_finish = output[-1][reward_var]
    wso_start = output[last_index_previous_state][reward_var]
    if wso_start is None: wso_start = 0.0
    if wso_finish is None: wso_finish = 0.0
    growth = wso_finish - wso_start
    if reward_var == "TWSO":  # hack to deal with different units
        growth = growth / 10.0
    current_date = output[-1]['day']
    previous_date = output[last_index_previous_state]['day']

    zero_nitrogen_results = zero_nitrogen_env_storage.get_episode_output(env_baseline)
    wso_current = zero_nitrogen_results[reward_var][current_date]
    wso_previous = zero_nitrogen_results[reward_var][previous_date]
    if wso_current is None: wso_current = 0.0
    if wso_previous is None: wso_previous = 0.0
    growth_baseline = wso_current - wso_previous
    if reward_var == "TWSO":  # hack to deal with different units
        growth_baseline = growth_baseline / 10.0

    benefits = growth - growth_baseline

    return benefits, growth


def n_demand_yield_reward(output, timestep, reward_var):
    assert 'TWSO' and 'Ndemand' in reward_var, f"reward_var does not contain TWSO and Ndemand"

    last_index_previous_state = (np.ceil(len(output) / timestep) - 1).astype('int') * timestep - 1
    n_demand_finish = output[-1][reward_var]['Ndemand']
    n_demand_start = output[last_index_previous_state][reward_var]['Ndemand']
    if n_demand_start is None: n_demand_start = 0.0
    if n_demand_finish is None: n_demand_finish = 0.0
    n_demand = n_demand_start - n_demand_finish

    wso_finish = output[-1][reward_var]['TWSO']
    wso_start = output[last_index_previous_state][reward_var]['TWSO']
    if wso_start is None: wso_start = 0.0
    if wso_finish is None: wso_finish = 0.0
    growth = wso_finish - wso_start
    if reward_var == "TWSO":  # hack to deal with different units
        growth = growth / 10.0

    benefits = growth - n_demand
    print(f"the N demand is {n_demand}")
    print(f"the benefits are {benefits}")

    return benefits, growth


def compute_economic_reward(wso, fertilizer, price_yield_ton=400.0, price_fertilizer_ton=300.0):
    g_m2_to_ton_hectare = 0.01
    convert_wso = g_m2_to_ton_hectare * price_yield_ton
    convert_fert = g_m2_to_ton_hectare * price_fertilizer_ton
    return 0.001 * (convert_wso * wso - convert_fert * fertilizer)


class ZeroNitrogenEnvStorage():
    """
    Container to store results from zero nitrogen policy (for re-use)
    """
    def __init__(self):
        self.results = {}

    def run_episode(self, env):
        env.reset()
        terminated, truncated = False, False
        infos_this_episode = []
        while not terminated or truncated:
            _, _, terminated, truncated, info = env.step(0)
            infos_this_episode.append(info)
        variables = infos_this_episode[0].keys()
        episode_info = {}
        for v in variables:
            episode_info[v] = {}
        for v in variables:
            for info_dict in infos_this_episode:
                episode_info[v].update(info_dict[v])
        return episode_info

    def get_key(self, env):
        year = env.date.year
        location = env._location
        return f'{year}-{location}'

    def get_episode_output(self, env):
        key = self.get_key(env)
        if key not in self.results.keys():
            results = self.run_episode(env)
            self.results[key] = results
        return self.results[key]