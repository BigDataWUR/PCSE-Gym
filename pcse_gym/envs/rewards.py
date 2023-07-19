import numpy as np


class Rewards:
    def __init__(self, reward_var, timestep, costs_nitrogen=10.0, zero_container=None):
        self.reward_var = reward_var
        self.timestep = timestep
        self.zero_container = zero_container
        self.costs_nitrogen = costs_nitrogen

    def last_index(self, output):
        return (np.ceil(len(output) / self.timestep).astype('int') - 1) * self.timestep - 1

    def zero_nitrogen_growth(self, output):
        current_date = output[-1]['day']
        previous_date = output[self.last_index(output)]['day']

        zero_nitrogen_results = self.zero_container.get_result
        wso_current = zero_nitrogen_results[self.reward_var][current_date]
        wso_previous = zero_nitrogen_results[self.reward_var][previous_date]
        if wso_current is None: wso_current = 0.0
        if wso_previous is None: wso_previous = 0.0
        growth_baseline = wso_current - wso_previous
        if self.reward_var == "TWSO":  # hack to deal with different units
            growth_baseline = growth_baseline / 10.0
        return growth_baseline

    def storage_organ_growth(self, output):

        wso_start = output[self.last_index(output)][self.reward_var]
        wso_finish = output[-1][self.reward_var]
        if wso_start is None: wso_start = 0.0
        if wso_finish is None: wso_finish = 0.0
        growth = wso_finish - wso_start
        if self.reward_var == "TWSO":  # hack to deal with different units
            growth = growth / 10.0
        return growth

    def default_winterwheat_reward(self, output, amount):

        growth = self.storage_organ_growth(output)

        growth_baseline = self.storage_organ_growth(output)

        benefits = growth - growth_baseline

        costs = self.costs_nitrogen * amount
        reward = benefits - costs

        return reward, growth

    # TODO agronomic nitrogen use efficiency still needs to be tested (See Vanlauwe et al, 2011)
    def ane_reward(self, output, amount):
        # agronomic nitrogen use efficiency
        growth = self.storage_organ_growth(output)

        growth_baseline = self.storage_organ_growth(output)

        benefits = growth - growth_baseline

        costs = amount * self.costs_nitrogen

        if amount == 0.0:  # avoid zero division
            amount = 1.0

        reward = benefits / amount - costs

        return reward, growth

    # TODO nitrogen use efficiency reward; WIP
    def nue_reward(self, output, amount):
        assert 'NuptakeTotal' and 'NLossesTotal' and 'NfixTotal' in \
               self.reward_var, f"reward_var does not contain NuptakeTotal, NLossesTotal or NfixTotal"

        n_upt = output[-1][self.reward_var]
        if n_upt is None: n_upt = 0.0
        n_loss = output[-1][self.reward_var]
        n_fix = output[-1][self.reward_var]

        fert = amount  # *costs_nitrogen

        crop_output = n_upt + n_loss

        crop_input = n_fix + fert

        nue = crop_output / crop_input

        return nue

    # TODO create reward surrounding crop N demand; WIP
    def n_demand_yield_reward(self, output, amount):
        assert 'TWSO' and 'Ndemand' in self.reward_var, f"reward_var does not contain TWSO and Ndemand"

        n_demand_finish = output[-1][self.reward_var]['Ndemand']
        n_demand_start = output[self.last_index(output)][self.reward_var]['Ndemand']
        if n_demand_start is None: n_demand_start = 0.0
        if n_demand_finish is None: n_demand_finish = 0.0
        n_demand = n_demand_start - n_demand_finish

        growth = self.storage_organ_growth(output)

        benefits = growth - n_demand
        print(f"the N demand is {n_demand}")
        print(f"the benefits are {benefits}")

        return benefits, growth


def compute_economic_reward(wso, fertilizer, price_yield_ton=400.0, price_fertilizer_ton=300.0):
    g_m2_to_ton_hectare = 0.01
    convert_wso = g_m2_to_ton_hectare * price_yield_ton
    convert_fert = g_m2_to_ton_hectare * price_fertilizer_ton
    return 0.001 * (convert_wso * wso - convert_fert * fertilizer)
