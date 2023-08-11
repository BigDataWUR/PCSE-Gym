import pcse_gym.utils.process_pcse_output as process_pcse


class Rewards:
    def __init__(self, reward_var, timestep, costs_nitrogen=10.0):
        self.reward_var = reward_var
        self.timestep = timestep
        self.costs_nitrogen = costs_nitrogen

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