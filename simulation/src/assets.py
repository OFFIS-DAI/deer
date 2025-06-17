"""
Definition of assets controlled by Flexibility Agent.
@author: Malin Radtke
"""
from logic.src.asset_representations import BatteryStorage


class SimulatedBatteryStorage(BatteryStorage):
    """
    Definition of a battery storage as a flexible asset.
    """

    def __init__(self, asset_id, capacity, max_power_w, min_power_w, f_start, p_start, efficiency_charge,
                 efficiency_discharge,
                 investment_costs_euro, number_of_full_cycles, is_pre_charging_storage=False, minimum_soc=0,
                 maximum_soc=1):
        super().__init__(asset_id, capacity, max_power_w, min_power_w, f_start, p_start, efficiency_charge,
                         efficiency_discharge,
                         investment_costs_euro, number_of_full_cycles, is_pre_charging_storage, minimum_soc,
                         maximum_soc)
        self.power_values = {0: 0}

    def get_flexibility(self, num_intervals, number_of_provisioning_intervals, aggregation_type=None, baseline_soc=None,
                        baseline_power=None, external_discharge=None, temp_env=None):
        pass


    def get_feasible_power_values(self, t_start, target_power) -> list:
        """
        Generate a list of feasible power values.
        @param t_start: timestamp of interval.
        @param target_power: target power to be achieved with all agents in pool.
        @param fixed_power_values: fixed power values due to other obligations.
        @return: list of feasible power values.
        """
        last_known_time, soc_at_last_known_time = self.get_last_known_soc_value(t_start=t_start)

        time_since_last_known_soc = (t_start - last_known_time) / 3600
        interval_length = 0.25 + time_since_last_known_soc

        print('Time since last known SoC: ', time_since_last_known_soc)

        power_options = []

        if target_power < 0:
            # Calculate the maximum SOC that can be achieved within this interval
            max_soc = min(1, soc_at_last_known_time + (
                    abs(self.max_power_w) * self.efficiency_charge * interval_length) / self.capacity)

            for factor in range(1, 50):
                # Divide target power to create feasible steps for charging
                power = target_power / factor
                effective_power = power * self.efficiency_charge  # Adjust for charging efficiency
                updated_soc = soc_at_last_known_time + (abs(effective_power) * interval_length) / self.capacity
                if updated_soc <= max_soc:
                    power_options.append(power)

        elif target_power > 0:
            # Calculate the minimum SOC that can be reached within this interval
            min_soc = max(0, soc_at_last_known_time - (
                    abs(self.min_power_w) / self.efficiency_discharge * interval_length) / self.capacity)

            for factor in range(1, 50):
                # Divide target power to create feasible steps for discharging
                power = target_power / factor
                effective_power = power / self.efficiency_discharge  # Adjust for discharging efficiency
                updated_soc = soc_at_last_known_time - (effective_power * interval_length) / self.capacity
                if updated_soc >= min_soc:
                    power_options.append(power)
        else:
            power_options = [0]

        # Return the list of feasible power values
        return power_options

