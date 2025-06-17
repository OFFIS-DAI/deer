import math
import numpy as np


def get_pre_charging_storage_flexibility_bands(battery_storage,
                                               num_intervals: int,
                                               provisioning_intervals: int,
                                               baseline_power: list,
                                               baseline_soc: list,
                                               bat_to_ev_values: list,
                                               grid_to_ev_values: list):
    """
    Calculates flexibility bands for pre-charging storages, considering EV charging demands.

    @param baseline_power: power values in battery baseline [W].
    @param battery_storage: battery storage object.
    @param num_intervals: number of intervals to calculate flex (15 min intervals assumed).
    @param provisioning_intervals: number of provisioning intervals (length of each flex calculation).
    @param baseline_soc: baseline SoC values.
    @param bat_to_ev_values: power drawn from battery to EV [W].
    @param grid_to_ev_values: power drawn from grid to EV [W].
    @return: flexibility as dict (per interval with min/max power).
    """
    # Extend the baseline power values as we are looking into the future
    baseline_power = [-p for p in baseline_power]
    assert len(baseline_power) >= num_intervals
    assert len(baseline_soc) >= num_intervals
    assert len(bat_to_ev_values) >= num_intervals
    assert len(grid_to_ev_values) >= num_intervals

    flexibilities = []

    # Iterate over each interval
    for interval in range(num_intervals):
        start_soc = baseline_soc[interval]

        capacity = battery_storage.capacity
        efficiency_charge = battery_storage.efficiency_charge
        max_charge = battery_storage.max_power_w

        # as this is a pre-charging storage, we do not have discharge (p_max) flexibility
        # therefore, we do not need the min SoC direction
        # we also never have discharge values in our baseline

        # Initialize SOC bounds
        current_max_soc = start_soc
        minP = []
        max_soc_values = [start_soc]

        if not np.all([p == 0 for p in bat_to_ev_values[interval:interval + provisioning_intervals]]):
            minP = [0] * provisioning_intervals

        else:
            # Forward pass: Calculate SOC bounds for each interval
            for interval_no in range(provisioning_intervals):
                baseline_ch = baseline_power[interval + interval_no] if baseline_power[
                                                                            interval + interval_no] > 0 else 0

                if max_charge + baseline_ch < 0:
                    raise ValueError('should be charge')
                else:
                    # Charging
                    max_soc_values.append(min(1, current_max_soc + (
                            max_charge * efficiency_charge) * 0.25 / capacity))
                current_max_soc = max_soc_values[-1]

            # Backward pass: Adjust SOC bounds to ensure future baseline power requirements are met
            current_max_soc = 1

            for interval_no in range(provisioning_intervals - 1, -1, -1):
                cur_baseline_charge = baseline_power[interval + interval_no] * efficiency_charge \
                    if baseline_power[interval + interval_no] > 0 \
                    else 0

                current_max_soc = min(1, current_max_soc - 0.25 / capacity * cur_baseline_charge)
                max_soc_values[interval_no] = min(max_soc_values[interval_no], current_max_soc)

            # Use the adjusted SOC bounds to calculate flexibility
            for interval_no in range(provisioning_intervals):
                max_soc_first = max_soc_values[interval_no]
                max_soc_second = max_soc_values[interval_no + 1]

                # Calculate power flexibility for this interval
                p_max_charge = ((max_soc_second - max_soc_first) * capacity / 0.25) * (1 / efficiency_charge)

                # Flip sign for discharge (positive) and charge (negative)
                minP.append(math.floor(-p_max_charge) if p_max_charge > 0 else 0)

        # Append flexibilities for the current interval
        flexibilities.append({
            'minP': minP,
            'maxP': [0] * provisioning_intervals
        })

    return flexibilities


def get_battery_flexibility_bands(battery_storage,
                                  num_intervals: int,
                                  provisioning_intervals: int,
                                  baseline_power: list,
                                  baseline_soc: list):
    """
    Extended calculation of flexibility bands with energy values and per provisioning interval.
    Ensures that flexibility in earlier intervals respects the need to meet baseline requirements in future intervals.
    @param baseline_power: power values in battery baseline [W].
    @param battery_storage: battery storage object.
    @param num_intervals: number of intervals to calculate flex (15 min intervals assumed).
    @param provisioning_intervals: number of provisioning intervals (length of each flex calculation).
    @param baseline_soc: baseline SoC values.
    @return: flexibility as dict (per interval with min/max power).
    """
    # Extend the baseline power values as we are looking into the future
    baseline_power = [-p for p in baseline_power]
    assert len(baseline_power) >= num_intervals
    assert len(baseline_soc) >= num_intervals
    flexibilities = []

    # Iterate over each interval
    for interval in range(num_intervals):
        start_soc = baseline_soc[interval]

        capacity = battery_storage.capacity
        efficiency_discharge = battery_storage.efficiency_discharge
        efficiency_charge = battery_storage.efficiency_charge
        max_charge = battery_storage.max_power_w
        max_discharge = battery_storage.min_power_w

        # Initialize SOC bounds
        current_min_soc = start_soc
        current_max_soc = start_soc

        minP = []
        maxP = []

        min_soc_values = [start_soc]
        max_soc_values = [start_soc]

        # Forward pass: Calculate SOC bounds for each interval
        for interval_no in range(provisioning_intervals):
            baseline_ch = baseline_power[interval + interval_no] if baseline_power[interval + interval_no] > 0 else 0
            baseline_dis = baseline_power[interval + interval_no] if baseline_power[interval + interval_no] < 0 else 0

            if max_discharge + baseline_dis < 0:
                # Discharging
                min_soc_values.append(max(battery_storage.minimum_soc, current_min_soc + (
                        max_discharge / efficiency_discharge) * 0.25 / capacity))
            else:
                raise ValueError('should be discharge')

            if max_charge + baseline_ch < 0:
                raise ValueError('should be charge')
            else:
                # Charging
                max_soc_values.append(min(battery_storage.maximum_soc, current_max_soc + (
                        max_charge * efficiency_charge) * 0.25 / capacity))
            current_max_soc = max_soc_values[-1]
            current_min_soc = min_soc_values[-1]

        # Backward pass: Adjust SOC bounds to ensure future baseline power requirements are met
        current_min_soc = battery_storage.minimum_soc
        current_max_soc = battery_storage.maximum_soc

        for interval_no in range(provisioning_intervals - 1, -1, -1):
            cur_baseline_charge = baseline_power[interval + interval_no] * efficiency_charge if baseline_power[
                                                                                                    interval + interval_no] > 0 \
                else 0
            cur_baseline_discharge = baseline_power[interval + interval_no] / efficiency_discharge if \
                baseline_power[interval + interval_no] < 0 \
                else 0
            current_max_soc = min(battery_storage.maximum_soc, current_max_soc - 0.25 / capacity * cur_baseline_charge)
            current_min_soc = max(battery_storage.minimum_soc, current_min_soc - 0.25 / capacity * cur_baseline_discharge)
            max_soc_values[interval_no] = min(max_soc_values[interval_no], current_max_soc)
            min_soc_values[interval_no] = max(min_soc_values[interval_no], current_min_soc)

        # Use the adjusted SOC bounds to calculate flexibility
        for interval_no in range(provisioning_intervals):
            min_soc_first = min_soc_values[interval_no]
            min_soc_second = min_soc_values[interval_no + 1]

            max_soc_first = max_soc_values[interval_no]
            max_soc_second = max_soc_values[interval_no + 1]

            # Calculate power flexibility for this interval
            p_max_discharge = ((min_soc_second - min_soc_first) * capacity / 0.25) * efficiency_discharge
            p_max_charge = ((max_soc_second - max_soc_first) * capacity / 0.25) * (1 / efficiency_charge)

            # Flip sign for discharge (positive) and charge (negative)
            minP.append(math.floor(-p_max_charge) if p_max_charge > 0 else 0)
            maxP.append(math.ceil(-p_max_discharge) if p_max_discharge < 0 else 0)

        # Append flexibilities for the current interval
        flexibilities.append({
            'minP': minP,
            'maxP': maxP
        })

    return flexibilities


def get_heatpump_flexibility_bands(heatpump,
                                   num_intervals: int,
                                   provisioning_intervals: int,
                                   baseline_power: list,
                                   baseline_soc: list,
                                   heat_demand: list,
                                   ambient_temp: list,
                                   interval_length: float = 0.25):
    """
    Calculates flexibility bands for hp. Power values can be different in each provisioning interval.
    Ensures that flexibility in earlier intervals respects the need to meet heat demand in future intervals.
    Note: both minP and maxP will have negative values!
    @param interval_length: length of the interval for which the flexibility has to be provided.
    @param heatpump: heat pump object.
    @param num_intervals: number of intervals to calculate flex (15 min intervals assumed).
    @param provisioning_intervals: number of provisioning intervals (length of each flex calculation).
      @param baseline_power: power values in battery baseline [W].
    @param baseline_soc: stored energy in hot water tank in kWh.
    @param heat_demand: aggregated space heating and domestic hot water demand.
    @param ambient_temp: temperature of environment
    @return: flexibility as dict.
    """

    assert len(baseline_soc) >= num_intervals and len(heat_demand) >= num_intervals and len(
        ambient_temp) >= num_intervals
    flexibilities = []

    # Iterate over each interval
    for flex_start in range(num_intervals):
        start_soc = baseline_soc[flex_start]
        # Initialize SOC bounds
        current_min_soc = start_soc
        current_max_soc = start_soc
        curr_min_soc_req = start_soc
        min_soc_values = []
        max_soc_values = []
        minP = []
        min_p_req = []  # minimal p required to meet heat demand
        maxP = []

        # Forward pass: Calculate reachable socs, also considering heat demand
        for i in range(provisioning_intervals):
            # calculate current water temperature based on current soc to get the cop value
            current_max_temp = heatpump.calculate_current_water_temp(current_max_soc)
            cop_t_max = heatpump.calculate_cop(current_max_temp, ambient_temp[flex_start + i])

            max_soc_next_interval = min(1, current_max_soc + (
                    heatpump.max_power_w * cop_t_max - heat_demand[
                i]) * 0.25 / heatpump.energy_capacity_wh)  # maximum charge
            # soc that would be reached when not consuming electrical energy
            min_soc_next_interval = max(0, current_min_soc + (
                    - heat_demand[i] * 0.25 / heatpump.energy_capacity_wh))  # maximum charge
            max_soc_values.append(max_soc_next_interval)
            min_soc_values.append(min_soc_next_interval)
            current_max_soc = max_soc_values[-1]
            current_min_soc = min_soc_values[-1]

            curr_min_temp = heatpump.calculate_current_water_temp(curr_min_soc_req)
            cop_t = heatpump.calculate_cop(curr_min_temp, ambient_temp[flex_start + i])
            p_min_req = max(0, (
                    curr_min_soc_req * heatpump.energy_capacity_wh / 0.25 + heat_demand[flex_start + i]) / cop_t)
            p_min_possible = min(heatpump.max_power_w, p_min_req)
            curr_min_soc_req = curr_min_soc_req + (
                    p_min_possible * cop_t - heat_demand[flex_start + i]) * 0.25 * heatpump.energy_capacity_wh
            min_p_req.append(p_min_req)

        # Backward pass: Adjust SOC bounds to ensure future baseline power requirements are met
        current_min_soc = 0
        current_max_soc = 1
        min_soc_heat_bl = 0
        for i in range(provisioning_intervals - 1, -1, -1):
            cop_t_min = heatpump.calculate_cop(heatpump.min_temp, ambient_temp[flex_start + i])
            cop_t_max = heatpump.calculate_cop(heatpump.max_temp, ambient_temp[flex_start + i])
            # cop_t_min = heatpump.calculate_worst_case_cop(ambient_temp[flex_start + i])
            # cop_t_max = heatpump.calculate_worst_case_cop(ambient_temp[flex_start + i])
            # current_min_soc = current_min_soc + (heat_demand[flex_start + i] * 0.25 / heatpump.energy_capacity_wh)
            current_min_soc = max(0, current_min_soc + (
                    (baseline_power[flex_start + i] * cop_t_min + heat_demand[
                        flex_start + i]) * 0.25 / heatpump.energy_capacity_wh))
            # current_min_soc = max(0, current_min_soc + heat_demand[flex_start + i] * 0.25 / heatpump.energy_capacity_wh)
            # current_min_soc = max(0, current_min_soc + heat_demand[
            #     flex_start + i] * 0.25 / heatpump.energy_capacity_wh)
            current_max_soc = min(1, current_max_soc - (
                    (-baseline_power[flex_start + i] * cop_t_max - heat_demand[
                        flex_start + i]) * 0.25 / heatpump.energy_capacity_wh))

            min_soc_values[i] = max(min_soc_values[i], current_min_soc)
            max_soc_values[i] = min(max_soc_values[i], current_max_soc)

        # Use the adjusted SOC bounds to calculate flexibility
        for i in range(provisioning_intervals):
            min_soc_first = start_soc if i == 0 else min_soc_values[i - 1]
            min_soc_second = min_soc_values[i]

            max_soc_first = start_soc if i == 0 else max_soc_values[i - 1]
            max_soc_second = max_soc_values[i]

            current_min_temp = heatpump.calculate_current_water_temp(min_soc_first)
            current_max_temp = heatpump.calculate_current_water_temp(max_soc_first)
            cop_t_min = heatpump.calculate_cop(current_min_temp, ambient_temp[flex_start + i])
            cop_t_max = heatpump.calculate_cop(current_max_temp, ambient_temp[flex_start + i])
            # Calculate power flexibility for this interval
            min_p = ((min_soc_second - min_soc_first) * heatpump.energy_capacity_wh / cop_t_min) / 0.25 + min_p_req[i]
            max_p = ((max_soc_second - max_soc_first) * heatpump.energy_capacity_wh / cop_t_max) / 0.25 + min_p_req[i]
            # flex is increased by heat demand, e.g.
            # max_p = min(heatpump.max_power_w + baseline_power[i],
            #             (((max_soc_second - max_soc_first) * heatpump.energy_capacity_wh / 0.25) + heat_demand[
            #                 i]) / cop_t_max)
            # Flip sign for discharge (positive) and charge (negative)
            # Flip sign for discharge (positive) and charge (negative)
            minP.append(math.ceil(min_p))
            maxP.append(math.ceil(-max_p))
            # Append flexibilities for the current interval
        flexibilities.append({
            'minP': maxP,
            # 'maxP': [-1 * p for p in min_p_req]
            'maxP': minP,
            # 'negP': maxP,
            # 'posP': minP, # reduction of energy consumption
        })

    return flexibilities
