import pcse_gym.utils.process_pcse_output as process_pcse
import numpy as np


def reward_functions_without_baseline():
    return ['GRO', 'DEP', 'ENY', 'NUE', 'HAR', 'NUP']


def reward_functions_with_baseline():
    return ['DEF', 'ANE', 'END']


def reward_functions_end():
    return ['END', 'ENY']


class Rewards:
    def __init__(self, reward_var, timestep, costs_nitrogen=10.0, vrr=0.7):
        self.reward_var = reward_var
        self.timestep = timestep
        self.costs_nitrogen = costs_nitrogen
        self.vrr = vrr

    def growth_storage_organ(self, output, amount, multiplier=1):
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
        costs = self.costs_nitrogen * amount
        reward = growth - costs
        return reward, growth

    def growth_reward_var(self, output, amount):
        growth = process_pcse.compute_growth_var(output, self.timestep, self.reward_var)
        costs = self.costs_nitrogen * amount
        reward = growth - costs
        return reward, growth

    def default_winterwheat_reward(self, output, output_baseline, amount, multiplier=1):
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
        growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)
        benefits = growth - growth_baseline
        costs = self.costs_nitrogen * amount
        reward = benefits - costs
        return reward, growth

    def deployment_reward(self, output, amount, multiplier=1, vrr=None):
        """
        reward function that mirrors a realistic (financial) cost of DT deployment in a field
        one unit of reward equals the price of 1kg of wheat yield
        """
        # recovered_fertilizer = amount * vrr
        # unrecovered_fertilizer = (amount - recovered_fertilizer) * self.various_costs()['environmental']
        if amount == 0:
            cost_deployment = 0
        else:
            cost_deployment = self.various_costs()['to_the_field']

        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
        # growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep)
        # fertilizer_price = self.various_costs()['fertilizer'] * amount
        costs = (self.costs_nitrogen * amount) + cost_deployment
        reward = growth - costs
        return reward, growth

    # agronomic nitrogen use efficiency (ee Vanlauwe et al, 2011)
    def ane_reward(self, ane_obj, output, output_baseline, amount):
        # agronomic nitrogen use efficiency
        reward, growth = ane_obj.reward(output, output_baseline, amount)
        return reward, growth

    def end_reward(self, end_obj, output, output_baseline, amount, multiplier=1):
        end_obj.calculate_cost_cumulative(amount)
        end_obj.calculate_positive_reward_cumulative(output, output_baseline)
        reward = 0 - amount * self.costs_nitrogen
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

        return reward, growth

    def nue_reward(self, nue_obj, output, output_baseline, amount, multiplier=1):
        nue_obj.calculate_cost_cumulative(amount)
        nue_obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)
        reward = 0 - amount * self.costs_nitrogen
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

        return reward, growth

    def calc_misc_cost(self, end_obj, cost):
        end_obj.calculate_misc_cumulative_cost(cost)

    # TODO create reward surrounding crop N demand; WIP
    def n_demand_yield_reward(self, output, multiplier=1):
        assert 'TWSO' and 'Ndemand' in self.reward_var, f"reward_var does not contain TWSO and Ndemand"
        n_demand_diff = process_pcse.compute_growth_var(output, self.timestep, 'Ndemand')
        growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
        benefits = growth - n_demand_diff
        print(f"the N demand is {n_demand_diff}")
        print(f"the benefits are {benefits}")
        return benefits, growth

    """
    Classes that determine the reward function
    """
    class DEF:
        """
        Relative yield reward function, as implemented in Kallenberg et al (2023)
        """
        def __init__(self, timestep, costs_nitrogen):
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)
            benefits = growth - growth_baseline
            costs = self.costs_nitrogen * amount
            reward = benefits - costs
            return reward, growth

    class GRO:
        """
        Absolute growth reward function, modified from Kallenberg et al. (2023)
        """
        def __init__(self, timestep, costs_nitrogen):
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            costs = self.costs_nitrogen * amount
            reward = growth - costs
            return reward, growth

    class DEP:
        """
        Reward function that considers a realistic (financial) cost of DT deployment in a field
        one unit of reward equals the price of 1kg of wheat yield
        """
        def __init__(self, timestep, costs_nitrogen):
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            if amount == 0:
                cost_deployment = 0
            else:
                cost_deployment = self.various_costs()['to_the_field']

            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            # growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep)
            # fertilizer_price = self.various_costs()['fertilizer'] * amount
            costs = (self.costs_nitrogen * amount) + cost_deployment
            reward = growth - costs
            return reward, growth

        @staticmethod
        def various_costs():
            return dict(
                to_the_field=10,
                fertilizer=1,
                environmental=2
            )

    class END:
        """
        Sparse reward function, modified from Kallenberg et al. (2023)
        Only provides positive reward at harvest
        """
        def __init__(self, timestep, costs_nitrogen):
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_cost_cumulative(amount)
            obj.calculate_positive_reward_cumulative(output, output_baseline)
            reward = 0 - amount * self.costs_nitrogen
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

            return reward, growth

    class NUE:
        """
        Sparse reward based on calculated nitrogen use efficiency
        """
        def __init__(self, timestep, costs_nitrogen):
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_cost_cumulative(amount)
            obj.calculate_amount(amount)
            obj.calculate_positive_reward_cumulative(output, output_baseline, multiplier)
            reward = 0 - amount * self.costs_nitrogen
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

            return reward, growth

    class NUP:
        """
        Reward based on Nitrogen Uptake, from Gautron et al. (2023)
        """
        def __init__(self, timestep, costs_nitrogen):
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            growth = process_pcse.compute_growth_var(output, self.timestep, 'NuptakeTotal')
            costs = self.costs_nitrogen * amount
            reward = growth - costs
            return reward, growth

    class HAR:
        """
        Sparse reward based on Wu et al. (2021) considering N losses
        """
        def __init__(self, timestep, costs_nitrogen, threshold=200, loss_modifier=1, penalty_modifier=1):
            self.timestep = timestep
            self.threshold = threshold
            self.costs_nitrogen = costs_nitrogen
            self.loss_modifier = loss_modifier
            self.penalty_modifier = penalty_modifier

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            # N application (N_t)
            obj.calculate_cost_n(amount)
            # N loss (N_l_t)
            n_loss = process_pcse.compute_growth_var(output, self.timestep, 'NLOSSCUM')
            obj.calculate_n_loss(n_loss)
            # Yield growth (Y)
            obj.calculate_positive_reward_cumulative(output)
            # Threshold
            penalty = obj.calculate_threshold(amount, self.threshold)

            reward = 0 - amount * self.costs_nitrogen - n_loss * self.loss_modifier - penalty * self.penalty_modifier
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

            return reward, growth

    """
    Containers for certain reward functions
    """
    class ContainerEND:
        """
        Container to keep track of cumulative positive rewards for end of timestep
        """
        def __init__(self, timestep, costs_nitrogen=10.0):
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

            self.cum_growth = .0
            self.cum_amount = .0
            self.cum_positive_reward = .0
            self.cum_cost = .0
            self.cum_misc_cost = .0
            self.cum_leach = .0

        def reset(self):
            self.cum_growth = .0
            self.cum_amount = .0
            self.cum_positive_reward = .0
            self.cum_cost = .0
            self.cum_misc_cost = .0
            self.cum_leach = .0

        def growth_storage_organ_wo_cost(self, output, multiplier=1):
            return process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)

        def default_winterwheat_reward_wo_cost(self, output, output_baseline, multiplier=1):
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)
            benefits = growth - growth_baseline
            return benefits

        def calculate_cost_cumulative(self, amount):
            self.cum_amount += amount
            self.cum_cost += amount * self.costs_nitrogen

        def calculate_misc_cumulative_cost(self, cost):
            self.cum_misc_cost += cost

        def calculate_positive_reward_cumulative(self, output, output_baseline=None, multiplier=1):
            if not output_baseline:
                benefits = self.growth_storage_organ_wo_cost(output, multiplier)
            else:
                benefits = self.default_winterwheat_reward_wo_cost(output, output_baseline, multiplier)
            self.cum_positive_reward += benefits

        def calculate_cost_n(self, amount):
            self.cum_amount += amount

        def calculate_n_loss(self, n_loss):
            self.cum_leach += n_loss

        def calculate_threshold(self, amount, threshold):
            if amount == 0:
                return 0
            else:
                return self.cum_amount - threshold

        @property
        def dump_cumulative_positive_reward(self) -> float:
            return self.cum_positive_reward

        @property
        def dump_cumulative_cost(self) -> float:
            return self.cum_cost + self.cum_misc_cost

    class ContainerNUE(ContainerEND):
        '''
        Container to keep track of rewards based on nitrogen use efficiency
        '''
        def __init__(self, timestep, costs_nitrogen=10.0):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen
            self.actions = 0

        def calculate_amount(self, action):
            self.actions += action

        def calculate_reward_nue(self, n_input, n_output, year=None):
            nue = calculate_nue(n_input, n_output, year=year)
            end_yield = super().dump_cumulative_positive_reward

            return unimodal_function(nue) * end_yield

        def reset(self):
            super().reset()
            self.actions = 0

    # ane_reward object
    class ContainerANE:
        """
        A container to keep track of the cumulative ratio of kg grain / kg N
        """
        def __init__(self, timestep):
            self.timestep = timestep
            self.cum_growth = 0
            self.cum_baseline_growth = 0
            self.cum_amount = 0
            self.moving_ane = 0

        def reward(self, output, output_baseline, amount):
            growth = self.cumulative(output, output_baseline, amount)
            benefit = self.cum_growth - self.cum_baseline_growth

            if self.cum_amount == 0.0:
                ane = benefit / 1.0
            else:
                ane = benefit / self.cum_amount
                self.moving_ane = ane
            ane -= amount  # TODO need to add environmental penalty and reward ANE that favours TWSO
            return ane, growth

        def cumulative(self, output, output_baseline, amount, multiplier=1):
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)

            self.cum_growth += growth
            self.cum_baseline_growth += growth_baseline
            self.cum_amount += amount
            return growth

        def reset(self):
            self.cum_growth = 0
            self.cum_baseline_growth = 0
            self.cum_amount = 0


class DummyClass:
    def reset(self):
        pass


def calculate_nue(n_input, n_so, year=None, n_seed=3.5):
    n_in = input_nue(n_input, year, n_seed=n_seed)
    nue = n_so / n_in
    return nue


def input_nue(n_input, year=None, n_seed=3.5):
    nh4, no3 = get_deposition_amount(year)
    n_depo = nh4 + no3
    return n_input + n_seed + n_depo


def get_deposition_amount(year) -> tuple:
    if year is None:
        NO3 = 12.5
        NH4 = 12.5
    else:
        ''' Linear functions of N deposition based on
            data in the Netherlands from CLO (2022)'''
        NO3 = 538.868 - 0.264 * year
        NH4 = 697 - 0.339 * year

    return NH4, NO3


def get_surplus_n(n_input, n_so, year=None, n_seed=3.5):
    n_i = input_nue(n_input, year=year, n_seed=n_seed)

    return n_i - n_so


#  piecewise conditions
def unimodal_function(b):
    """
    For NUE reward, coefficient indicating how close the NUE in the range of 70-90%
    """
    if b < 0.7:
        return 0.9 * np.exp(-10 * (0.7 - b)) + 0.1
    elif 0.7 <= b <= 0.9:
        return 1
    else:  # b > 0.9
        return 0.9 * np.exp(-10 * (b - 0.9)) + 0.1

def compute_economic_reward(wso, fertilizer, price_yield_ton=400.0, price_fertilizer_ton=300.0):
    g_m2_to_ton_hectare = 0.01
    convert_wso = g_m2_to_ton_hectare * price_yield_ton
    convert_fert = g_m2_to_ton_hectare * price_fertilizer_ton
    return 0.001 * (convert_wso * wso - convert_fert * fertilizer)


