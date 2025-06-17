import math


def get_valid_battery_flexibility(battery_storage,
                                  num_intervals: int,
                                  provisioning_intervals: int,
                                  baseline_power: list,
                                  baseline_soc: list):
    """
    Calculation of flexibility bands for valid flex-offers under assumption of baseline.
    Ensures a constant power value is calculated for the entire provisioning interval.

    Sign convention:
    - Discharge (power flowing out of battery): positive (+)
    - Charge (power flowing into battery): negative (-)

    @param battery_storage: battery storage object with properties:
                           - capacity: battery capacity in Wh
                           - efficiency_discharge: discharge efficiency (0-1)
                           - efficiency_charge: charge efficiency (0-1)
                           - max_power_w: maximum charging power (positive value)
                           - min_power_w: minimum discharging power (negative value)
    @param num_intervals: number of intervals to calculate flexibility for (15 min intervals assumed)
    @param provisioning_intervals: number of intervals for each flexibility calculation (constant flexibility)
    @param baseline_power: baseline power values (+ discharge, - charge)
    @param baseline_soc: baseline SoC values (0-1)
    @return: flexibility as list of dicts with minP and maxP values
    """
    assert len(baseline_soc) >= num_intervals + provisioning_intervals - 1, "Insufficient baseline_soc values"
    assert len(baseline_power) >= num_intervals + provisioning_intervals - 1, "Insufficient baseline_power values"

    # Extract battery parameters
    capacity = battery_storage.capacity  # Wh
    eff_dis = battery_storage.efficiency_discharge  # (0-1)
    eff_ch = battery_storage.efficiency_charge  # (0-1)
    max_ch_power = -battery_storage.max_power_w  # Max charging power (negative)
    max_dis_power = -battery_storage.min_power_w  # Max discharging power (positive)

    flexibilities = []

    # Time step in hours
    dt = 0.25  # 15 minutes

    # For each starting interval
    for interval in range(num_intervals):
        start_soc = baseline_soc[interval]

        baseline_power_in_interval = baseline_power[interval:interval + provisioning_intervals]

        dis_energy_in_storage = start_soc * capacity
        ch_energy_in_storage = - (1 - start_soc) * capacity

        e_baseline_ch = sum([p * dt * eff_ch if p < 0 else 0 for p in
                             baseline_power_in_interval])
        e_baseline_dis = sum([p * dt * (1 / eff_dis) if p > 0 else 0 for p in
                              baseline_power_in_interval])

        available_energy_ch = min(0, ch_energy_in_storage - e_baseline_ch)
        available_energy_dis = max(0, dis_energy_in_storage - e_baseline_dis)

        available_power_ch = available_energy_ch / (dt*provisioning_intervals * eff_ch)
        available_power_dis = available_energy_dis / (dt * provisioning_intervals) * eff_ch

        max_baseline_power_ch = min(baseline_power_in_interval)
        max_baseline_power_dis = max(baseline_power_in_interval)

        max_ch_p = math.ceil(max(max_ch_power - max_baseline_power_ch, available_power_ch))
        max_dis_p = math.floor(min(max_dis_power - max_baseline_power_dis, available_power_dis))

        # Add to results
        flexibilities.append({
            'minP': max_ch_p,  # Additional charging (negative)
            'maxP': max_dis_p  # Additional discharging (positive)
        })

    return flexibilities
