import pcse_gym.utils.process_pcse_output as process_pcse


class Rewards:
    def __init__(self, reward_var, timestep, costs_nitrogen=10.0, vrr=0.7):
        self.reward_var = reward_var
        self.timestep = timestep
        self.costs_nitrogen = costs_nitrogen
        self.vrr = vrr

    def growth_storage_organ(self, output, amount):
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep)
        costs = self.costs_nitrogen * amount
        reward = growth - costs
        return reward, growth

    def growth_reward_var(self, output, amount):
        growth = process_pcse.compute_growth_var(output, self.timestep, self.reward_var)
        costs = self.costs_nitrogen * amount
        reward = growth - costs
        return reward, growth

    def default_winterwheat_reward(self, output, output_baseline, amount):
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep)
        growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep)
        benefits = growth - growth_baseline
        costs = self.costs_nitrogen * amount
        reward = benefits - costs
        return reward, growth

    # TODO reward based on cost if deploying in the field, to be tested; WIP
    def deployment_reward(self, output, output_baseline, amount, vrr):
        """
        reward function that mirrors a realistic (financial) cost of DT deploymeny in a field
        one unit of reward/cost equals roughly $1
        """
        recovered_fertilizer = amount * vrr
        unrecovered_fertilizer = (amount - recovered_fertilizer) * various_costs()['environmental']
        cost_deployment = various_costs()['deployment']

        growth = process_pcse.compute_growth_storage_organ(output, self.timestep)
        growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep)
        benefits = growth - growth_baseline
        fertilizer_price = various_costs()['fertilizer'] * amount
        costs = fertilizer_price + unrecovered_fertilizer + cost_deployment
        reward = benefits - costs
        return reward, growth

    # TODO agronomic nitrogen use efficiency still needs to be tested (See Vanlauwe et al, 2011)
    def ane_reward(self, output, output_baseline, amount):
        # agronomic nitrogen use efficiency
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep)
        growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep)
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
    def n_demand_yield_reward(self, output):
        assert 'TWSO' and 'Ndemand' in self.reward_var, f"reward_var does not contain TWSO and Ndemand"
        n_demand_diff = process_pcse.compute_growth_var(output, self.timestep, 'Ndemand')
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep)
        benefits = growth - n_demand_diff
        print(f"the N demand is {n_demand_diff}")
        print(f"the benefits are {benefits}")
        return benefits, growth


def compute_economic_reward(wso, fertilizer, price_yield_ton=400.0, price_fertilizer_ton=300.0):
    g_m2_to_ton_hectare = 0.01
    convert_wso = g_m2_to_ton_hectare * price_yield_ton
    convert_fert = g_m2_to_ton_hectare * price_fertilizer_ton
    return 0.001 * (convert_wso * wso - convert_fert * fertilizer)

@staticmethod
def various_costs():
    return dict(
        deployment=50,
        fertilizer=1,
        environmental=2
    )
