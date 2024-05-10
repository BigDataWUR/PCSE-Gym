import pcse_gym.utils.process_pcse_output as process_pcse
import numpy as np

from abc import ABC, abstractmethod
import datetime
import calendar


def reward_functions_without_baseline():
    return ['GRO', 'DEP', 'ENY', 'NUE', 'HAR', 'NUP']


def reward_functions_with_baseline():
    return ['DEF', 'ANE', 'END']


def reward_function_list():
    return ['DEF', 'GRO', 'DEP', 'ENY', 'NUE', 'DNU', 'HAR', 'NUP', 'END', 'FIN']


def reward_functions_end():
    return ['END', 'ENY']


class Rewards:
    def __init__(self, reward_var, timestep, costs_nitrogen=10.0, vrr=0.7, with_year=False):
        self.reward_var = reward_var
        self.timestep = timestep
        self.costs_nitrogen = costs_nitrogen
        self.vrr = vrr
        self.profit = 0
        self.with_year = with_year

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

    def reset(self):
        self.profit = 0

    def calculate_profit(self, output, amount, year, multiplier, with_year=False, country='NL'):
        
        profit, _ = calculate_net_profit(output, amount, year, multiplier, self.timestep, with_year=with_year, country=country)

        return profit

    def update_profit(self, output, amount, year, multiplier, country='NL'):
        self.profit += self.calculate_profit(output, amount, year, multiplier, with_year=self.with_year)

    """
    Classes that determine the reward function
    """

    class Rew(ABC):
        def __init__(self, timestep, costs_nitrogen):
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        @abstractmethod
        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            raise NotImplementedError

    class DEF(Rew):
        """
        Relative yield reward function, as implemented in Kallenberg et al (2023)
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep, multiplier)
            benefits = growth - growth_baseline
            costs = self.costs_nitrogen * amount
            reward = benefits - costs
            return reward, growth

    class GRO(Rew):
        """
        Absolute growth reward function, modified from Kallenberg et al. (2023)
        """

        def __init__(self, timestep, costs_nitrogen):
            super().__init__(timestep, costs_nitrogen)
            self.timestep = timestep
            self.costs_nitrogen = costs_nitrogen

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep, multiplier)
            costs = self.costs_nitrogen * amount
            reward = growth - costs
            return reward, growth


    class FIN(Rew):
        """
        Financial reward function, converting yield, N fertilizer and labour costs into a net profit reward.
        """
        def __init__(self, timestep, costs_nitrogen, labour=False):
            super().__init__(timestep, costs_nitrogen)
            self.labour = labour
            self.base_labour_cost_index = 28.9  # euros, in 2020
            self.time_per_hectare = 5 / 60  # minutes to hours
            self.country = 'NL'

        def return_reward(self, output, amount, output_baseline=None, multiplier=1, obj=None):
            obj.calculate_amount(amount)

            year = process_pcse.get_year_in_step(output)

            reward, growth = calculate_net_profit(output, amount, year, multiplier, self.timestep, with_year=False)

            return reward, growth



class ActionsContainer:
    def __init__(self):
        self.actions = 0

    def calculate_amount(self, action):
        self.actions += action

    def reset(self):
        self.actions = 0

    @property
    def get_total_fertilization(self):
        return self.actions


def get_days_in_year(year):
    return 365 + calendar.isleap(year)


def compute_economic_reward(wso, fertilizer, price_yield_ton=400.0, price_fertilizer_ton=300.0):
    g_m2_to_ton_hectare = 0.01
    convert_wso = g_m2_to_ton_hectare * price_yield_ton
    convert_fert = g_m2_to_ton_hectare * price_fertilizer_ton
    return 0.001 * (convert_wso * wso - convert_fert * fertilizer)


def calculate_net_profit(output, amount, year, multiplier, timestep, with_year=False, with_labour=False, country='NL'):

    '''Get growth of Crop'''
    growth = process_pcse.compute_growth_storage_organ(output, timestep, multiplier)

    '''Convert growth to wheat price in the year'''
    wso_conv_eur = growth * get_wheat_price_in_kgs(year, with_year=with_year)

    '''Convert price of used fertilizer in the year'''
    n_conv_eur = get_fertilizer_price(amount, year, with_year=with_year)

    '''Convert labour price based on year'''
    labour_conv_eur = get_labour_price(year, with_labour=with_labour)

    '''Flag for fertilization action'''
    labour_flag = 1 if amount else 0

    reward = wso_conv_eur - n_conv_eur - labour_conv_eur * labour_flag

    return reward, growth


def annual_price_wheat_per_ton(year):
    prices = {
        1989: 177.16, 1990: 168.27, 1991: 174.05, 1992: 171.61, 1993: 148.94, 1994: 135.27, 1995: 131.89,
        1996: 130.50, 1997: 120.84, 1998: 111.39, 1999: 111.62, 2000: 116.23, 2001: 112.17, 2002: 102.89,
        2003: 114.73, 2004: 116.95, 2005: 96.73, 2006: 117.95, 2007: 180.78, 2008: 169.84, 2009: 112.23,
        2010: 152.00, 2011: 197.5, 2012: 219.28, 2013: 203.23, 2014: 164.12, 2015: 159.43, 2016: 145.17,
        2017: 154.62, 2018: 176.23, 2019: 172.23, 2020: 181.67, 2021: 233.84, 2022: 312.56, 2023: 227.56
    }

    return prices[year]


def get_wheat_price_in_kgs(year, with_year=False, price_per_ton=157.75):
    if not with_year:
        return price_per_ton * 0.001
    return annual_price_wheat_per_ton(year) * 0.001


def get_nitrogen_price_in_kgs(year, with_year=False, price_per_quintal=20.928):
    if not with_year:
        return price_per_quintal * 0.01
    return annual_price_nitrogen_per_quintal(year) * 0.01


def annual_price_nitrogen_per_quintal(year):
    prices = {
        1989: 11.61, 1990: 11.61, 1991: 12.20, 1992: 11.04, 1993: 10.07, 1994: 10.24, 1995: 12.58,
        1996: 13.22, 1997: 11.49, 1998: 10.55, 1999: 9.48, 2000: 13.09, 2001: 15.60, 2002: 14.28,
        2003: 15.18, 2004: 15.89, 2005: 17.11, 2006: 18.85, 2007: 19.81, 2008: 33.12, 2009: 21.37,
        2010: 21.71, 2011: 29.39, 2012: 29.38, 2013: 27.13, 2014: 27.74, 2015: 27.85, 2016: 21.49,
        2017: 21.37, 2018: 22.90, 2019: 24.17, 2020: 20.49, 2021: 35.71, 2022: 76.62, 2023: 38.14
    }

    return prices[year]


def labour_index_per_year(year):
    """
    Linear function to estimate hourly labour costs per year in the Netherlands
    From https://ycharts.com/indicators/netherlands_labor_cost_index
    """
    index = 2.0016 * year - 3941.4

    index = index / 100  # convert to percentage
    return index


"""
Calculations for getting prices in the year
"""


def get_fertilizer_price(action, year, with_year=False):
    """
    Price of N fertilizer per kg in the year

    :param action: agent's action
    :param year: year of the action
    :return: nitrogen price per kg
    """
    amount = action * 10  # action to kg/ha
    price = get_nitrogen_price_in_kgs(year, with_year)

    return amount * price


def get_labour_price(year, base_labour_cost_index=28.9, time_per_hectare=0.0834, with_labour=False):
    """
    Price of hourly labour per year, considering the European labour cost index

    :param base_labour_cost_index: labour cost in the base year of the index
    :param time_per_hectare: assumption of the time needed to fertilize one hectare of land, currently defaults to
            5 minutes per hectare.
    :return: price of labour in euros
    """

    if with_labour:
        return 0

    return (base_labour_cost_index * labour_index_per_year(year) + base_labour_cost_index) * time_per_hectare
