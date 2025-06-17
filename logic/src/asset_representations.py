"""
Definition of assets controlled by Flexibility Agent.
@author: Malin Radtke
"""
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Asset(ABC):
    """
    Abstract definition of an asset.
    """

    def __init__(self,
                 asset_id: str,
                 description: str):
        self.asset_id = asset_id
        self.description = description

    @abstractmethod
    def setup(self):
        pass


class FlexibleAsset(Asset, ABC):
    """
    Abstract definition of a flexible asset (can provide flexibility).
    """

    def __init__(self, asset_id: str, description: str):
        super().__init__(asset_id, description)

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_flexibility(self, num_intervals, number_of_provisioning_intervals, aggregation_type=None, baseline_soc=None,
                        baseline_power=None, external_discharge=None, temp_env=None):
        """
        Abstract method. Child class should implement and optimize flexibility offer.
        @param baseline_power:
        @param baseline_soc: SoC values in baseline schedule.
        @param num_intervals:
        @param aggregation_type:
        @param number_of_provisioning_intervals:
        @param external_discharge: demand that has to be met (sh + dhw for heat-pumps or charging schedules in car charging stations)
        @param temp_env
        @return:
        """
        pass


class FixedAsset(Asset, ABC):
    """
    Abstract definition of a non-flexible asset (cannot provide flexibility).
    """

    def __init__(self, description: str):
        asset_id = description
        super().__init__(asset_id, description)

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_forecast(self, current_timestamp: float, forecast_length: int, integrate_noise=True):
        """
        Abstract method. Child class should implement and return forecast for asset.
        @param integrate_noise:
        @param current_timestamp:
        @param forecast_length
        @return:
        """
        pass


class BatteryStorage(FlexibleAsset, ABC):
    """
    Definition of a battery storage as a flexible asset.
    """

    def __init__(self, asset_id, capacity, max_power_w, min_power, f_start, p_start,
                 efficiency_charge, efficiency_discharge,
                 investment_costs_euro, number_of_full_cycles, is_pre_charging_storage=False,
                 minimum_soc=0, maximum_soc=1):
        super().__init__(asset_id=asset_id, description='BatteryStorage')
        self.discharging_costs_euro_per_Wh = None
        self.power_values = None
        self.charging_costs_euro_per_Wh = None
        self.soc_values = None
        self.minimum_soc = minimum_soc
        self.maximum_soc = maximum_soc
        self.full_cycle_equivalents_pd = None
        self.p_start = p_start
        self.capacity = capacity
        self.max_power_w = max_power_w
        self.min_power_w = min_power
        self.f_start = f_start
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.investment_costs_euro = investment_costs_euro
        self.number_of_full_cycles = number_of_full_cycles
        self.is_pre_charging_storage = is_pre_charging_storage

        self.setup()

    def setup(self):
        self.full_cycle_equivalents_pd = None

        self.power_values = {}  # {timestamp: power value from this time on until current time or next key}
        self.soc_values = {0: self.f_start}  # {timestamp: SoC at timestamp}

        self.charging_costs_euro_per_Wh, self.discharging_costs_euro_per_Wh = self.calculate_cost_values()

    def reset_f_start(self, f_start):
        self.f_start = f_start
        self.soc_values = {0: f_start}  # {timestamp: SoC at timestamp}

    def calculate_cost_values(self) -> Tuple[float, float]:
        """
        Calculates costs for charging and discharging in currency (€) per power (W).
        @return: costs for charging, costs for discharging
        """
        costs_full_cycle_ch = (self.investment_costs_euro /
                               (2 * self.number_of_full_cycles * self.capacity * (1 / self.efficiency_charge)))
        costs_full_cycle_dis = (self.investment_costs_euro /
                                (2 * self.number_of_full_cycles * self.capacity * self.efficiency_discharge))

        return costs_full_cycle_ch, costs_full_cycle_dis

    def get_last_known_soc_value(self, t_start):
        """
        Identify last known SoC value from timestamp t_start.
        @param t_start: current timestamp.
        @return: (timestamp of last known SoC value, SoC value)
        """
        # Find the last known SoC before the requested time
        last_known_time = max([t for t in self.soc_values if t <= t_start], default=None)

        if last_known_time is None:
            print("Error: No previous SoC value available.")
            return False

        # Get the SoC at the last known time
        soc_at_last_known_time = self.soc_values[last_known_time]
        return last_known_time, soc_at_last_known_time

    def calculate_soc_for_timestamp(self, timestamp_of_interval_start):
        """
        Calculate the current State of Charge (SoC) based on the latest known SoC value
        and any active power settings that have occurred since then.

        @param timestamp_of_interval_start: The timestamp for which to calculate SoC
        @return: The calculated SoC value
        """
        # Find the latest SoC timestamp and value before the given time
        available_timestamps = [t for t in self.soc_values.keys() if t <= timestamp_of_interval_start]
        if not available_timestamps:
            # If no previous SoC values exist, return the default SoC
            return self.f_start

        latest_soc_timestamp = max(available_timestamps)
        latest_soc = self.soc_values[latest_soc_timestamp]

        # Get all power values between latest_soc_timestamp and current_time
        # Sort them to process in chronological order
        power_timestamps = sorted([t for t in self.power_values.keys()
                                   if latest_soc_timestamp <= t < timestamp_of_interval_start])

        # Start with the latest known SoC
        current_soc = latest_soc
        last_timestamp = latest_soc_timestamp

        # Process each power interval to update the SoC
        for timestamp in power_timestamps:
            power_value = self.power_values[timestamp]
            time_diff_h = (timestamp - last_timestamp) / 3600  # Convert seconds to hours

            # Calculate SoC change based on power value
            if power_value <= 0:  # Charging (negative power)
                # Use charging efficiency
                energy_delta_wh = abs(power_value) * time_diff_h * self.efficiency_charge
                soc_change = energy_delta_wh / self.capacity
                current_soc += soc_change  # Increase SoC when charging
            else:  # Discharging (positive power)
                # Use discharging efficiency
                energy_delta_wh = power_value * time_diff_h / self.efficiency_discharge
                soc_change = energy_delta_wh / self.capacity
                current_soc -= soc_change  # Decrease SoC when discharging

            # Update the last timestamp
            last_timestamp = timestamp

            # Ensure SoC stays within valid range [0, 1]
            current_soc = max(0.0, min(1.0, current_soc))

        # Process any remaining time between the last power timestamp and current_time
        if last_timestamp < timestamp_of_interval_start:
            # If there's a gap between the last power setting and current time,
            # we assume the last power value continues
            if power_timestamps:
                # Use the most recent power value
                power_value = self.power_values[power_timestamps[-1]]
                time_diff_h = (timestamp_of_interval_start - last_timestamp) / 3600

                if power_value <= 0:  # Charging
                    energy_delta_wh = abs(power_value) * time_diff_h * self.efficiency_charge
                    soc_change = energy_delta_wh / self.capacity
                    current_soc += soc_change
                else:  # Discharging
                    energy_delta_wh = power_value * time_diff_h / self.efficiency_discharge
                    soc_change = energy_delta_wh / self.capacity
                    current_soc -= soc_change

                # Ensure SoC stays within valid range
                current_soc = max(0.0, min(1.0, current_soc))

        return current_soc

    def power_request_is_feasible(self, power_request, t_start) -> bool:
        """
        Check if power request power_request for time interval t_start is feasible.
        @param power_request: request.
        @param t_start: timestamp.
        @return: True if feasible, else False.
        """
        if t_start not in self.soc_values.keys():
            # Get the last known SOC and corresponding power value
            self.soc_values[t_start] = self.calculate_soc_for_timestamp(t_start)

        # Convert SOC from percentage to actual energy available in Wh
        ch_energy_available_wh = (1 - self.soc_values[t_start]) * self.capacity
        dis_energy_available_wh = self.soc_values[t_start] * self.capacity

        # Calculate the energy required to meet the power request over 15 minutes (0.25 hours)
        time_interval_hours = 0.25
        if power_request > 0:
            # Discharge scenario
            energy_required_wh = power_request * time_interval_hours / self.efficiency_discharge
        else:
            # Charge scenario (negative power request), check if battery can absorb energy
            energy_required_wh = abs(power_request) * time_interval_hours * self.efficiency_charge

        # Check if the energy available is enough to supply the requested power
        if power_request > 0:
            # Check if the battery can provide the power (discharging)
            if dis_energy_available_wh >= energy_required_wh or abs(dis_energy_available_wh - energy_required_wh) < 5:
                self.power_values[t_start] = power_request
                return True
            print('Energy required = ', energy_required_wh, ', energy available: ', dis_energy_available_wh)
            print('SoC values: ', self.soc_values, ' and time to fix power: ', t_start)
            return False
        else:
            # Check if the battery can charge the power
            if ch_energy_available_wh >= energy_required_wh or abs(ch_energy_available_wh - energy_required_wh) < 5:
                self.power_values[t_start] = power_request
                return True
            print('Energy required = ', energy_required_wh, ', energy available: ', ch_energy_available_wh)
            print('SoC values: ', self.soc_values, ' and time to fix power: ', t_start)
            return False