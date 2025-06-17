import math

import numpy as np
from pyomo.core import Objective, minimize, value, ConstraintList, inequality, Constraint
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Var,
    Reals,  # type: ignore
    NonNegativeReals,  # type: ignore
    Binary,  # type: ignore
)
import time

from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# from simulation.src.definitions.assets import BatteryStorage, HeatPump
from logic.src.asset_representations import BatteryStorage, HeatPump


def get_pyo_value(component):
    """Retrieve the value of a Pyomo Variable or Parameter in a consistent way."""
    if hasattr(component, 'value'):
        return component.value
    return component  # Return raw value if it's not a Pyomo object


def optimize(num_intervals: int, time_interval_size: float, fixed_consumption: list, fixed_feed_in: list,
             prices_consumption_cent_per_wh: list,
             prices_feed_in_cent_per_wh: list,
             bat: BatteryStorage = None,
             f_start_bat: float = None,
             heat_pump: HeatPump = None,
             f_start_hp: float = None,
             heat_demand: list = None,
             ambient_temperature: list = None,
             fixed_power_values: list = None,
             absolute_power_tolerance: float = 5,
             fixed_consumption_through_bat = [],
             neglect_min_max_soc: bool = False
             ):
    """
    Args:
        num_intervals (int): Total number of intervals to optimize.
        time_interval_size (list): Size of each interval in hours.
        fixed_consumption ():  aggregated fixed power consumption in household, e.g. household demand
        fixed_feed_in (): aggregated fixed power feed in from household, e.g. PV gen
        prices_consumption_cent_per_wh (list): Prices for power consumption.
        prices_feed_in_cent_per_wh (list): Prices for power feed-in.
        bat (): Battery object with properties.
        f_start_bat (): start SOC of battery storage (may be different from battery object due to batches)
        heat_pump (): Heat Pump object with properties.
        f_start_hp (): start SOC of heat pump storage (may be different to heat pump object due to batches)
        heat_demand (): heat demand of household (hot water + heating) - needed if HP is included
        ambient_temperature (): ambient temperature of the house - needed if HP is included
        fixed_power_values ():  (list, optional): Predefined power values (from called offers; -charge & +discharge).
        absolute_power_tolerance (float, optional): Tolerance for power optimization in W.

    Returns:

    """
    # all power parameters and variables are defined as positive
    # initialize model
    model = ConcreteModel()
    model.T = RangeSet(0, num_intervals - 1)

    # define all parameters and variables
    model.fixed_consumption = Param(model.T, initialize=fixed_consumption)
    if len(fixed_consumption_through_bat) == 0:
        fixed_consumption_through_bat = [0] * num_intervals
    model.fixed_consumption_through_bat = Param(model.T, initialize=fixed_consumption_through_bat)
    model.fixed_feed_in = Param(model.T, initialize=fixed_feed_in)
    model.prices_consumption_cent_per_wh = Param(model.T, initialize=prices_consumption_cent_per_wh)
    model.prices_feed_in_cent_per_wh = Param(model.T, initialize=prices_feed_in_cent_per_wh)
    max_s = max(fixed_consumption)
    min_s = -max(fixed_feed_in)

    if bat is not None:
        max_s += bat.max_power_w
        min_s += bat.min_power_w
        model.bat_p_charge = Var(model.T, within=NonNegativeReals, bounds=(0, bat.max_power_w),
                                 initialize=np.zeros(num_intervals))
        model.bat_p_discharge = Var(model.T, within=NonNegativeReals, bounds=(0, abs(bat.min_power_w)),
                                    initialize=np.zeros(num_intervals))
        if not neglect_min_max_soc:
            # check if f_start exceeds min and max soc
            if f_start_bat > bat.maximum_soc or f_start_bat < bat.minimum_soc:
                neglect_min_max_soc = True
        min_soc = bat.minimum_soc if not neglect_min_max_soc else 0
        max_soc = bat.maximum_soc if not neglect_min_max_soc else 1
        model.bat_soc = Var(model.T, within=NonNegativeReals, bounds=(min_soc, max_soc))
        model.is_charging = Var(model.T, within=Binary)

        if bat.is_pre_charging_storage:
            # Only define these variables for pre-charging storage
            model.bat_to_ev = Var(model.T, within=NonNegativeReals, initialize=np.zeros(num_intervals))
            model.grid_to_ev = Var(model.T, within=NonNegativeReals, initialize=np.zeros(num_intervals))
        else:
            # For regular batteries, define these as parameters with zero values
            model.bat_to_ev = Param(model.T, initialize=np.zeros(num_intervals))
            model.grid_to_ev = Param(model.T, initialize=np.zeros(num_intervals))

        @model.Constraint(model.T)
        def max_power_rule(m, t):
            return m.bat_p_charge[t] <= bat.max_power_w * m.is_charging[t]

        @model.Constraint(model.T)
        def min_power_rule(m, t):
            return m.bat_p_discharge[t] <= abs(bat.min_power_w) * (1 - m.is_charging[t])

        if bat.is_pre_charging_storage:
            @model.Constraint(model.T)
            def no_charge_with_fixed_consumption(m, t):
                if fixed_consumption_through_bat[t] > 0:
                    # When there's fixed consumption through battery, force is_charging to 0
                    return m.is_charging[t] == 0
                else:
                    return Constraint.Skip

            @model.Constraint(model.T)
            def ev_demand_fulfilled(m, t):
                # Total power to EV must equal the demand (from battery + from grid)
                return (m.bat_to_ev[t] + m.grid_to_ev[t]) == fixed_consumption_through_bat[t]

            @model.Constraint(model.T)
            def bat_to_ev_limit(m, t):
                if fixed_consumption_through_bat[t] == 0:
                    # Skip if no demand (already constrained to be 0 above)
                    return Constraint.Skip
                elif t == m.T.first():
                    # For the first interval with demand, use f_start_bat
                    max_available_power = ((f_start_bat - min_soc) * bat.capacity /
                                           time_interval_size * bat.efficiency_discharge)
                    return m.bat_to_ev[t] <= max_available_power
                else:
                    # For subsequent intervals with demand, use the previous SOC
                    max_available_power = ((m.bat_soc[t - 1] - min_soc) * bat.capacity /
                                           time_interval_size * bat.efficiency_discharge)
                    return m.bat_to_ev[t] <= max_available_power

        # State of charge definition
        @model.Constraint(model.T)
        def soc_definition_bat(m, t):
            # For regular batteries, m.bat_to_ev[t] will be 0
            if t == m.T.first():
                return (m.bat_soc[t] == f_start_bat +
                        (bat.efficiency_charge * m.bat_p_charge[t] - (
                                m.bat_p_discharge[t] + m.bat_to_ev[t]) / bat.efficiency_discharge)
                        * time_interval_size / bat.capacity)
            return (m.bat_soc[t] == m.bat_soc[t - 1] +
                    (bat.efficiency_charge * m.bat_p_charge[t] - (
                            m.bat_p_discharge[t] + m.bat_to_ev[t]) / bat.efficiency_discharge)
                    * time_interval_size / bat.capacity)

        @model.Constraint
        def cycle_limit_rule(m):
            return sum((m.bat_p_charge[t] + m.bat_p_discharge[t]) * time_interval_size
                       for t in m.T) / (2 * bat.capacity) <= 2

    else:
        model.bat_p_charge = Param(model.T, domain=NonNegativeReals, initialize=np.zeros(num_intervals))
        model.bat_p_discharge = Param(model.T, domain=NonNegativeReals, initialize=np.zeros(num_intervals))

        # For regular batteries, define these as parameters with zero values
        model.bat_to_ev = Param(model.T, initialize=np.zeros(num_intervals))
        model.grid_to_ev = Param(model.T, initialize=np.zeros(num_intervals))
    if heat_pump is not None:
        max_s += heat_pump.max_power_w
        model.hp_power = Var(model.T, within=NonNegativeReals, initialize=np.zeros(num_intervals),
                             bounds=(0, heat_pump.max_power_w))
        model.hp_soc = Var(model.T, within=NonNegativeReals, bounds=(0, 1))
        model.heat_demand = Param(model.T, initialize=heat_demand)

        model.cop = Param(model.T,
                          initialize=[heat_pump.calculate_worst_case_cop(temp) for temp in ambient_temperature])

        @model.Constraint(model.T)
        def soc_definition_hp(m, t):
            if t == m.T.first():
                return (m.hp_soc[t] == f_start_hp + (m.hp_power[t] * m.cop[t] - m.heat_demand[t])
                        * time_interval_size / heat_pump.energy_capacity_wh)
            return (m.hp_soc[t] == m.hp_soc[t - 1] + (m.hp_power[t] * m.cop[t] - m.heat_demand[t])
                    * time_interval_size / heat_pump.energy_capacity_wh)

        # avoid short cycling
        # auxiliary variable to count cold starts of the hp
        model.hp_starts = Var(model.T, within=Binary)
        # New auxiliary variable for the rate of power change
        model.delta_P = Var(model.T, within=NonNegativeReals)

        epsilon = 0.1
        model.hp_on = Var(model.T, within=Binary)

        @model.Constraint(model.T)
        def state_condition_upper(m, t):
            return m.hp_power[t] <= m.hp_on[t] * heat_pump.max_power_w  # if hp_on[t] = 0, hp_power[t] = 0

        @model.Constraint(model.T)
        def state_condition_lower(m, t):
            return m.hp_power[t] >= m.hp_on[t] * epsilon  # if hp_power[t] > epsilon, hp_on[t] = 1

        @model.Constraint(model.T)
        def start_condition(m, t):
            if t > 0:
                return m.hp_starts[t] >= model.hp_on[t] - model.hp_on[t - 1]
            else:
                return Constraint.Skip

        @model.Constraint(model.T)
        def delta_p_upper(m, t):
            if t > 0:
                return m.delta_P[t] >= m.hp_power[t] - m.hp_power[t - 1]
            else:
                return Constraint.Skip

        @model.Constraint(model.T)
        def delta_p_lower(m, t):
            if t > 0:
                return m.delta_P[t] >= m.hp_power[t - 1] - m.hp_power[t]
            else:
                return Constraint.Skip

    else:
        model.hp_power = Param(model.T, initialize=np.zeros(num_intervals))
        model.hp_soc = Param(model.T, initialize=np.zeros(num_intervals))
        model.hp_starts = Param(model.T, initialize=np.zeros(num_intervals))
        model.delta_P = Param(model.T, initialize=np.zeros(num_intervals))

    if fixed_power_values:
        # Initialize a ConstraintList in your model
        model.fixed_power_constraints = ConstraintList()

        # Define a constraint function with an allowable range
        def fixed_power_rule_approx(m, i, p):
            return inequality(p - absolute_power_tolerance,
                              m.bat_p_charge[i] + m.hp_power[i] - m.bat_p_discharge[i],
                              p + absolute_power_tolerance)

        # Add constraints to the model using a loop with approximate p-values
        for i, p in fixed_power_values:
            model.fixed_power_constraints.add(fixed_power_rule_approx(model, i, p))

    penalty = np.mean(prices_consumption_cent_per_wh)
    penalty = 1e12
    # Smoothing factor for changes in power (higher values = stronger punishment)
    C_smooth = 0.05

    # define objective and solve model
    def costs(m):
        return sum(
            # Add grid_to_ev to the consumption from grid
            ((m.fixed_consumption[t] + m.bat_p_charge[t] + m.hp_power[t] + m.grid_to_ev[t]) * time_interval_size) *
            m.prices_consumption_cent_per_wh[t] +
            ((m.fixed_feed_in[t] + m.bat_p_discharge[t]) * time_interval_size)
            * m.prices_feed_in_cent_per_wh[t]  # feed in price is negative
            + C_smooth * model.delta_P[t] + penalty * m.hp_starts[t]
            for t in m.T)

    model.objective = Objective(rule=costs, sense=minimize)
    opt = SolverFactory('cbc', solver_io="lp")
    opt.options['seconds'] = 10
    opt.options['ratiogap'] = 0.001  # Tighter optimality gap
    opt.options['primalT'] = 1e-8  # Tighter primal tolerance
    return try_optimize(model, opt)


def try_optimize(model, opt):
    try:
        results = opt.solve(model,  # tee=True, keepfiles=True
                            )
        if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal):
            # Do something when the solution is optimal and feasible

            return model
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            # Do something when model in infeasible
            return False
        elif results.solver.termination_condition == TerminationCondition.maxTimeLimit:
            last_ratio = opt.options['ratio']  # optimality gap as a fraction of objective value (e.g., 0.01 = 1% gap).
            last_primal_t = opt.options["primalT"]  # allowed constraint violation (default 1e-7)
            if last_ratio is None:
                last_ratio = 0
            next_ratio = last_ratio + 0.01  # increase the allowed optimality gap by 1 %
            if last_primal_t is None:
                last_primal_t = 1e-7
            next_primal_t = last_primal_t * 10
            opt.options['ratio'] = next_ratio
            if next_primal_t <= 1:
                opt.options["primalT"] = next_primal_t
            if next_ratio <= 0.1:
                print(f'retrying with optimality gap {next_ratio} and primalT {next_primal_t}')
                try_optimize(model, opt)
            else:
                print('Found no solution in time with maximum optimality gap')
                print('Solver Status: ', results.solver.status)
        else:
            # Something else is wrong
            print('Solver Status: ', results.solver.status)
        return model
    except Exception as ex:
        model.write("failed_model_debug.lp", io_options={"symbolic_solver_labels": True})
        print("Solver failed with exception:", ex)


def optimize_household_baseline(start_timestamp: float,
                                num_intervals: int,
                                prices_consumption_cent_per_wh: list,
                                prices_feed_in_cent_per_wh: list,
                                fixed_power: list,
                                fixed_consumption_through_bat = [],
                                time_interval_size: float = 0.25,
                                heat_demand: list = None,
                                ambient_temperature: list = None,
                                battery_storage: BatteryStorage = None,
                                heat_pump: HeatPump = None,
                                assumed_fixed_power_values=None,
                                absolute_power_tolerance=0,
                                batch_size=24):
    """
    Optimizes the baseline of a household with respect to costs
    Args:
        start_timestamp ():
        num_intervals (int): Total number of intervals to optimize.
        time_interval_size (list): Size of each interval in hours.
        prices_consumption_cent_per_wh (list): Prices for power consumption.
        prices_feed_in_cent_per_wh (list): Prices for power feed-in.
        fixed_power (): aggregated fixed power values in household, e.g. household demand and PV gen
        heat_demand (): heat demand of household (hot water + heating) - needed if HP is included
        ambient_temperature (): ambient temperature of the house - needed if HP is included
        battery_storage (): Battery object with properties.
        heat_pump (): Heat Pump object with properties.
        assumed_fixed_power_values (list, optional): Predefined power values (because of called offers; charge is negative and discharge is positive).
        absolute_power_tolerance (float, optional): Tolerance for power optimization in W.
        batch_size (int): Number of intervals to optimize per batch.

    Returns:
        tuple: Aggregated power values, total costs, and SoC values for the entire period.
    """
    if assumed_fixed_power_values:
        # now the signs are flipped, as the baseline optimization works with...
        # (-): discharge
        # (+): charge
        assumed_fixed_power_values = [(i, -p) for i, p in assumed_fixed_power_values if i < num_intervals]
        batch_size = assumed_fixed_power_values[-1][0] + 1
    else:
        assumed_fixed_power_values = []

    charge_values = []
    discharge_values = []
    p_hp = []
    costs = []
    p_grid_to_ev = []
    p_bat_to_ev = []

    # remember non-fulfilled fixed power values
    non_fulfilled_fixed_power_values = []

    if battery_storage is not None:
        f_start_bat = battery_storage.calculate_soc_for_timestamp(start_timestamp)
        soc_values_bat = [f_start_bat]
    else:
        soc_values_bat = [0]
        f_start_bat = 0
    if heat_pump is not None:
        soc_values_hp = [heat_pump.f_start]
    else:
        soc_values_hp = [0]

    # Divide optimization into batches
    for batch_start in range(0, num_intervals, batch_size):
        # Define the current batch
        batch_end = min(batch_start + batch_size, num_intervals)
        current_num_intervals = batch_end - batch_start

        # Slice data for the current batch
        current_fixed_power = fixed_power[batch_start:batch_end]
        current_prices_consumption = prices_consumption_cent_per_wh[batch_start:batch_end]
        current_prices_feed_in = prices_feed_in_cent_per_wh[batch_start:batch_end]
        if heat_pump is not None:
            current_heat_demand = heat_demand[batch_start:batch_end]
            current_ambient_temperature = ambient_temperature[batch_start:batch_end]
        else:
            current_heat_demand, current_ambient_temperature = None, None

        # Fixed power values for current batch
        current_fixed_power_values = [
            (i - batch_start, p) for i, p in assumed_fixed_power_values if batch_start <= i < batch_end
        ] if assumed_fixed_power_values else None

        # p>0 -> power demand (e.g., from household)
        fixed_consumption = [math.ceil(p) if p > 0 else 0 for p in current_fixed_power]
        # p<0 -> power feed-in (e.g., from pv plant)
        fixed_feed_in = [math.ceil(-p) if p < 0 else 0 for p in current_fixed_power]

        current_fixed_consumption_through_bat = [math.ceil(abs(p))
                                                 for p in fixed_consumption_through_bat[batch_start:batch_end]]

        solution_found = False
        neglect_min_max_soc = False
        while not solution_found:
            # Initialize and configure model for the current batch
            model = optimize(current_num_intervals, time_interval_size, fixed_consumption, fixed_feed_in,
                             current_prices_consumption,
                             current_prices_feed_in,
                             battery_storage,
                             f_start_bat=f_start_bat if batch_start == 0 else soc_values_bat[-1],
                             heat_pump=heat_pump,
                             f_start_hp=soc_values_hp[-1],
                             heat_demand=current_heat_demand,
                             ambient_temperature=current_ambient_temperature,
                             fixed_power_values=current_fixed_power_values,
                             absolute_power_tolerance=absolute_power_tolerance,
                             fixed_consumption_through_bat=current_fixed_consumption_through_bat,
                             neglect_min_max_soc=neglect_min_max_soc
                             )
            if not model:
                if assumed_fixed_power_values and len(current_fixed_power_values) > 0:
                    # get fixed power value with max key
                    last_fixed_power_value_i = max([index for index, _ in current_fixed_power_values])
                    non_fulfilled_fixed_power_values.extend([(i, v) for i, v in current_fixed_power_values
                                                             if i == last_fixed_power_value_i])
                    current_fixed_power_values = [(i, v) for i, v in current_fixed_power_values
                                                  if i != last_fixed_power_value_i]
                else:
                    absolute_power_tolerance += 1
                    neglect_min_max_soc = True
                    if absolute_power_tolerance >= 50:
                        return None
            else:
                solution_found = True

                # Append results from the current batch to the overall lists
                charge_values.extend([get_pyo_value(model.bat_p_charge[t]) for t in range(0, current_num_intervals)])
                discharge_values.extend(
                    [get_pyo_value(model.bat_p_discharge[t]) for t in range(0, current_num_intervals)])
                p_hp.extend([get_pyo_value(model.hp_power[t]) for t in range(0, current_num_intervals)])
                if battery_storage is not None:
                    soc_values_bat.extend([model.bat_soc[t].value for t in range(current_num_intervals)])
                if heat_pump is not None:
                    soc_values_hp.extend([model.hp_soc[t].value for t in range(current_num_intervals)])
                    print(f'hp baseline: {[model.hp_power[t].value for t in range(current_num_intervals)]}')
                    # print(f'hp on: {[model.hp_on[t].value for t in range(current_num_intervals)]}')
                    print(f'hp starts: {[model.hp_starts[t].value for t in range(current_num_intervals)]}')
                costs.append(value(model.objective))
                if battery_storage is not None and battery_storage.is_pre_charging_storage:
                    p_grid_to_ev.extend([model.grid_to_ev[t].value for t in range(current_num_intervals)])
                    p_bat_to_ev.extend([model.bat_to_ev[t].value for t in range(current_num_intervals)])
                else:
                    # Add zeros to maintain consistent list lengths
                    p_grid_to_ev.extend([0] * current_num_intervals)
                    p_bat_to_ev.extend([0] * current_num_intervals)

    # Return all interval data: aggregated charge/discharge values, total costs, and SoC values
    total_cost = sum(costs)
    # this way, charge is negative and discharge is positive
    p_bat = [math.ceil(discharge - charge) if (discharge-charge) < 0 else math.floor(discharge - charge)
             for charge, discharge in zip(charge_values, discharge_values)]
    if battery_storage:
        battery_storage.power_values[start_timestamp] = p_bat[0]

    p_hp = [-1 * p for p in p_hp]

    return (p_bat, p_grid_to_ev, p_bat_to_ev, p_hp, total_cost, soc_values_bat, soc_values_hp,
            non_fulfilled_fixed_power_values)
