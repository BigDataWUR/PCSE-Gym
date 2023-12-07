import pcse_gym.utils.process_pcse_output as process_pcse


def reward_functions_without_baseline():
    return ['GRO', 'DEP']

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
    def deployment_reward(self, output, amount, vrr=None):
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

        growth = process_pcse.compute_growth_storage_organ(output, self.timestep)
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

    @staticmethod
    def various_costs():
        return dict(
            to_the_field=10,
            fertilizer=1,
            environmental=2
        )

    #ane_reward object
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

        def cumulative(self, output, output_baseline, amount):
            growth = process_pcse.compute_growth_storage_organ(output, self.timestep)
            growth_baseline = process_pcse.compute_growth_storage_organ(output_baseline, self.timestep)

            self.cum_growth += growth
            self.cum_baseline_growth += growth_baseline
            self.cum_amount += amount
            return growth

        def reset(self):
            self.cum_growth = 0
            self.cum_baseline_growth = 0
            self.cum_amount = 0


def compute_economic_reward(wso, fertilizer, price_yield_ton=400.0, price_fertilizer_ton=300.0):
    g_m2_to_ton_hectare = 0.01
    convert_wso = g_m2_to_ton_hectare * price_yield_ton
    convert_fert = g_m2_to_ton_hectare * price_fertilizer_ton
    return 0.001 * (convert_wso * wso - convert_fert * fertilizer)


