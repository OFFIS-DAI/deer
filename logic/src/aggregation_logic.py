"""
Flexibility Optimization Module

This module implements optimization algorithms for aggregating and disaggregating flexibility offers 
from distributed energy resources. It uses the Pyomo optimization framework with the CBC solver
to perform the following key functions:

1. Aggregation of flexibility offers from multiple agents
2. Disaggregation of flexibility requests to multiple agents
3. Cost-optimal or best-effort distribution of flexibility requests

The module primarily deals with two types of flexibility:
- "min" flexibility: Reduction in power consumption
- "max" flexibility: Increase in power consumption
"""

import numpy as np
from pyomo.core import Objective, Constraint, maximize, minimize
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Var,
    Reals,  # type: ignore
    NonNegativeReals,  # type: ignore
    Binary,  # type: ignore
    NonPositiveReals,  # type: ignore
)
from pyomo.opt import TerminationCondition, SolverStatus, SolverFactory


def nl_optimize_bi(agent_flexibilities, offer_interval_length, flex_type="min",
                   virtual_lb=False, target_schedule=None, best_effort=False):
    """
    Performs bi-level optimization for flexibility aggregation or disaggregation.

    This function creates and solves an optimization model to either:
    1. Maximize/minimize the aggregated flexibility when target_schedule is None (aggregation)
    2. Distribute a target schedule among agents optimally (disaggregation)

    The optimization respects various constraints including:
    - Power bounds for each agent (min/max flexibility)
    - Energy transfer constraints across time intervals
    - Direction-specific flexibility constraints

    Args:
        agent_flexibilities (dict): Dictionary mapping agent IDs to their flexibility data.
            Each entry contains baseline_values, flex_values (with minP and maxP),
            and costs information.
        offer_interval_length (int): Number of time intervals to consider in the optimization.
        flex_type (str): Type of flexibility to optimize for. Options:
            - "min": Minimize power consumption (default)
            - "max": Maximize power consumption
        virtual_lb (bool): If True, allows virtual lower bounds based on baseline values.
        target_schedule (list, optional): Target flexibility values as deviation from baselines in W.
            If provided, performs disaggregation instead of aggregation.
        best_effort (bool): If True, tries to reach the target as closely as possible
            without considering costs. If False, optimizes costs within a small
            deviation from the target.

    Returns:
        tuple: A tuple containing:
            - float: Average aggregated flexibility value (or None if infeasible)
            - float: Total costs of the solution (or None if infeasible)
            - dict: Mapping of agent IDs to their optimal flexibility schedules (or None if infeasible)
    """
    # Create a concrete optimization model
    model = ConcreteModel()
    model.T = RangeSet(0, offer_interval_length - 1)

    # Extract agent IDs, excluding any aggregated baseline
    agents = list(agent_flexibilities.keys())
    if 'aggre_bl' in agents:
        agents.remove('aggre_bl')

    # Step 1: Add variables and parameters for each agent
    for agent, data in agent_flexibilities.items():
        if agent in agents:
            baseline = data["baseline_values"]
            flex_values = data["flex_values"]

            # Main power variable for each agent and time step
            setattr(model, f"P_{agent}", Var(model.T, within=Reals, initialize=0))

            # Determine upper bound and auxiliary variables based on flex_type
            if flex_type == "max":
                # For max flexibility, find the maximum between baseline and maxP
                ub = max([max(baseline[t], flex_values["maxP"][t]) for t in model.T])
                # Auxiliary variable for tracking power transfer (non-negative for max flex)
                setattr(model, f"p_tra_aux{agent}", Var(model.T, within=NonNegativeReals, initialize=0))
            else:
                # For min flexibility, find the minimum between baseline and minP
                ub = min([min(baseline[t], flex_values["minP"][t]) for t in model.T])
                # Auxiliary variable for tracking power transfer (non-positive for min flex)
                setattr(model, f"p_tra_aux{agent}", Var(model.T, within=NonPositiveReals, initialize=0))

            # Add flexibility parameters to the model
            setattr(model, f"flex_max_P_{agent}",
                    Param(model.T, initialize={t: flex_values["maxP"][t] for t in model.T})
                    )
            setattr(model, f"flex_min_P_{agent}",
                    Param(model.T, initialize={t: flex_values["minP"][t] for t in model.T})
                    )

            # Add baseline parameters to the model
            setattr(model, f"bl_{agent}",
                    Param(model.T, initialize={t: baseline[t] for t in model.T})
                    )
            # Add positive baseline component as a parameter
            setattr(model, f"bl_pos_{agent}",
                    Param(model.T, initialize={t: max(0, baseline[t]) for t in model.T})
                    )
            # Add negative baseline component as a parameter
            setattr(model, f"bl_neg_{agent}",
                    Param(model.T, initialize={t: min(0, baseline[t]) for t in model.T})
                    )

            # Upper bound parameter
            setattr(model, f"bound_{agent}", Param(initialize=ub))

            # Power transfer variable to track energy shifts between time intervals
            setattr(model, f"p_tra_{agent}", Var(model.T, within=Reals, initialize=0))

    # Step 2: Add constraints for each agent
    for agent in agents:
        # Power transfer constraint: tracks how much energy is shifted between time intervals
        def p_tra_rule(m, t):
            p_tra = getattr(m, f"p_tra_{agent}")
            if t == 0:
                return p_tra[t] == 0  # No energy transfer in first interval

            p_agent = getattr(m, f"P_{agent}")
            maxP = getattr(m, f"flex_max_P_{agent}")
            minP = getattr(m, f"flex_min_P_{agent}")
            bl_pos = getattr(m, f"bl_pos_{agent}")
            bl_neg = getattr(m, f"bl_neg_{agent}")

            # Calculate accumulated power transfer up to this time step
            p_aux = 0
            for k in range(0, t):
                if flex_type == "max":
                    # For max flexibility, transfer is based on difference from maxP
                    p_aux += maxP[k] - p_agent[k] - bl_pos[k]
                else:
                    # For min flexibility, transfer is based on difference from minP
                    p_aux += minP[k] - p_agent[k] - bl_neg[k]

            # Set the power transfer value for this time step
            return p_tra[t] == p_aux

        # Constraint to ensure power transfer is in the correct direction (max or min)
        def p_tra_dir_bound1(m, t):
            p_tra = getattr(m, f"p_tra_{agent}")
            p_tra_aux = getattr(m, f"p_tra_aux{agent}")

            if flex_type == "max":
                # For max flexibility, auxiliary variable must be <= p_tra
                return p_tra_aux[t] <= p_tra[t]
            else:
                # For min flexibility, auxiliary variable must be >= p_tra
                return p_tra_aux[t] >= p_tra[t]

        # Constraint to ensure auxiliary variable has correct sign
        def p_tra_dir_bound2(m, t, ):
            p_tra_aux = getattr(m, f"p_tra_aux{agent}")

            if flex_type == "max":
                # For max flexibility, auxiliary variable must be non-negative
                return p_tra_aux[t] >= 0
            else:
                # For min flexibility, auxiliary variable must be non-positive
                return p_tra_aux[t] <= 0

        # Add the power transfer constraints to the model
        setattr(model, f"p_tra_constraint_{agent}", Constraint(model.T, rule=p_tra_rule))
        setattr(model, f"p_tra_dir_constraint1_{agent}", Constraint(model.T, rule=p_tra_dir_bound1))
        setattr(model, f"p_tra_dir_constraint2_{agent}", Constraint(model.T, rule=p_tra_dir_bound2))

        # Maximum power constraint: ensures power stays within flexibility limits with transfer
        def max_power_rule(m, t, ):
            p_agent = getattr(m, f"P_{agent}")
            bl = getattr(m, f"bl_{agent}")
            p_tra = getattr(m, f"p_tra_aux{agent}")
            maxP = getattr(m, f"flex_max_P_{agent}")
            minP = getattr(m, f"flex_min_P_{agent}")

            if flex_type == "max":
                # For max flexibility, power must be <= max flexibility limit plus transfer
                return p_agent[t] <= maxP[t] - bl[t] + p_tra[t]
            else:
                # For min flexibility, power must be >= min flexibility limit plus transfer
                return p_agent[t] >= minP[t] - bl[t] + p_tra[t]

        # Minimum power constraint: sets lower or upper bounds based on flexibility type
        def min_power_rule(m, t, ):
            p_agent = getattr(m, f"P_{agent}")
            maxP = getattr(m, f"flex_max_P_{agent}")
            minP = getattr(m, f"flex_min_P_{agent}")
            bl_pos = getattr(m, f"bl_pos_{agent}")
            bl_neg = getattr(m, f"bl_neg_{agent}")

            if flex_type == "max":
                if virtual_lb:
                    # If virtual lower bound is enabled, power >= -positive baseline
                    return p_agent[t] >= -bl_pos[t]
                else:
                    # Otherwise, power must be non-negative
                    return p_agent[t] >= 0
            else:
                if virtual_lb:
                    # If virtual lower bound is enabled, power <= -negative baseline
                    return p_agent[t] <= -bl_neg[t]
                else:
                    # Otherwise, power must be non-positive
                    return p_agent[t] <= 0

        # Absolute maximum power constraint: ensures total power stays within bounds
        def absolute_max_power_rule(m, t):
            p_agent = getattr(m, f"P_{agent}")
            bl = getattr(m, f"bl_{agent}")
            upper_b = getattr(m, f"bound_{agent}")

            if flex_type == "max":
                # For max flexibility, total power must be <= upper bound
                return p_agent[t] + bl[t] <= upper_b
            else:
                # For min flexibility, total power must be >= upper bound (which is a minimum)
                return p_agent[t] + bl[t] >= upper_b

        # Add all power constraint rules to the model
        setattr(
            model,
            f"max_power_constraint_{agent}",
            Constraint(model.T, rule=max_power_rule)
        )

        setattr(
            model,
            f"min_power_constraint_{agent}",
            Constraint(model.T, rule=min_power_rule)
        )

        setattr(
            model,
            f"absolute_max_power_constraint_{agent}",
            Constraint(model.T, rule=absolute_max_power_rule)
        )

    # For aggregation (no target_schedule), add constraint to ensure constant flexibility
    if not target_schedule:
        @model.Constraint(model.T)
        def p_tick_rule(m, t):
            if t < len(m.T) - 1:
                # Sum of all agent powers at t must equal sum at t+1 (constant flexibility)
                return sum(getattr(m, f"P_{a}")[t] for a in agents) == sum(
                    getattr(m, f"P_{a}")[t + 1] for a in agents)
            return Constraint.Skip

    # Define the aggregation objective function
    def aggregation_objective(m):
        # Sum of all agent powers across all time intervals
        return sum(sum(getattr(m, f"P_{a}")[t] for t in m.T) for a in agents)

    # If target_schedule is provided, set up disaggregation model
    if target_schedule is not None:
        # Add target schedule as parameter
        model.target = Param(model.T, initialize={t: target_schedule[t] for t in model.T})

        # Add auxiliary variable to measure absolute deviation from target
        model.delta = Var(model.T, within=NonNegativeReals)

        # Constraints to calculate absolute deviation |actual - target|
        def abs_constraint_1(m, t):
            # delta >= actual - target
            return m.delta[t] >= sum(getattr(m, f"P_{a}")[t] for a in agents) - m.target[t]

        def abs_constraint_2(m, t):
            # delta >= target - actual
            return m.delta[t] >= m.target[t] - sum(getattr(m, f"P_{a}")[t] for a in agents)

        model.abs_constr_1 = Constraint(model.T, rule=abs_constraint_1)
        model.abs_constr_2 = Constraint(model.T, rule=abs_constraint_2)

        if not best_effort:
            # For cost optimization with target, limit deviation to epsilon
            epsilon = 1

            # Maximum deviation constraint
            def epsilon_deviation_rule(m, t):
                return m.delta[t] <= epsilon

            model.epsilon_constraint = Constraint(model.T, rule=epsilon_deviation_rule)

            # Objective: minimize costs while staying within epsilon of target
            def disaggregation_objective(m):
                if flex_type == "max":
                    # For max flexibility, use discharge costs
                    return sum(
                        sum(getattr(m, f"P_{a}")[t] * agent_flexibilities[agent]["costs_dis"] for a in agents) for t in
                        m.T)
                else:
                    # For min flexibility, use charging costs (note the negative sign)
                    return sum(
                        sum(-getattr(m, f"P_{a}")[t] * agent_flexibilities[agent]["costs_ch"] for a in agents) for t in
                        m.T)

            model.objective = Objective(rule=disaggregation_objective, sense=minimize)
        else:
            # For best effort, ignore costs and minimize total deviation from target
            def target_deviation(m):
                return sum(m.delta[t] for t in m.T)

            model.objective = Objective(rule=target_deviation, sense=minimize)
    else:
        # For aggregation, maximize or minimize total flexibility based on flex_type
        if flex_type == "max":
            model.objective = Objective(rule=aggregation_objective, sense=maximize)
        else:
            model.objective = Objective(rule=aggregation_objective, sense=minimize)

    # Solve the optimization model using CBC solver
    opt = SolverFactory('cbc')
    results = opt.solve(model)  # Use tee=True for verbose output

    # Process the results
    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        # Solution is optimal and feasible
        agent_schedules = {}

        # Determine costs based on flexibility type
        if flex_type == "min":
            fixed_costs = agent_flexibilities[agent]["costs_ch"]
        elif flex_type == "max":
            fixed_costs = agent_flexibilities[agent]["costs_dis"]

        # Calculate costs and extract schedules
        costs = 0
        for a in agents:
            # Get the optimal power schedule for each agent
            agent_schedules[a] = [getattr(model, f"P_{a}")[t].value for t in model.T]
            # Calculate costs (assuming 15-minute intervals = 0.25 hours)
            costs += abs(sum(agent_schedules[a])) * 0.25 * fixed_costs

        # Calculate the aggregated flexibility
        aggregated_flex = [sum(getattr(model, f"P_{a}")[t].value for a in agents) for t in model.T]

        # Return the mean flexibility, costs, and agent schedules
        return np.mean(aggregated_flex), costs, agent_schedules
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        # Model is infeasible, no solution found
        print('no solution found')
        return None, None, None
    else:
        # Other solver status
        print('Solver Status: ', results.solver.status)
        return None, None, None


def get_relevant_flex(agent_flexibilities, offer_interval_length, flex_interval=None, flex_index=None):
    """
    Filters flexibility data for the required interval and ensures feasibility.

    This function extracts the relevant portion of flexibility data for a specific time interval
    and corrects any inconsistencies between flexibility limits and baseline values that might
    lead to infeasible optimization models.

    Args:
        agent_flexibilities (dict): Dictionary of agent flexibility data.
        offer_interval_length (int): Number of time intervals to consider.
        flex_interval (int, optional): Timestamp for the start of the flexibility interval.
        flex_index (int, optional): Index of the flexibility interval relative to agent data start.
            Either flex_interval or flex_index must be provided.

    Returns:
        dict: Dictionary of feasible flexibility data for the specified interval.

    Raises:
        Exception: If neither flex_interval nor flex_index is specified.
    """
    feasible_flex = {}

    for agent, data in agent_flexibilities.items():
        # Determine the flexibility index if not provided
        if flex_index is None:
            if flex_interval is not None:
                # Calculate index based on timestamp (assuming 15-minute intervals)
                flex_index = int((flex_interval - data.t_start) / (15 * 60))
            else:
                raise Exception(f"Either Flex index or flex Interval must be specified")

        # Extract baseline values for the relevant interval
        baseline = data.baseline_values[flex_index:flex_index + offer_interval_length]

        # Extract flexibility values
        flex_values = data.flex_values[flex_index]

        # Store the data in the result dictionary
        feasible_flex[agent] = {
            'baseline_values': baseline,
            'costs_ch': data.costs_ch,
            'costs_dis': data.costs_dis,
            'flex_values': flex_values
        }

        # Ensure flexibility limits are consistent with baseline
        for t in range(len(flex_values['minP'])):
            # If minP is greater than baseline, set it equal to baseline
            if flex_values['minP'][t] > baseline[t]:
                feasible_flex[agent]["flex_values"]['minP'][t] = baseline[t]

            # If maxP is less than baseline, set it equal to baseline
            if flex_values['maxP'][t] < baseline[t]:
                feasible_flex[agent]["flex_values"]['maxP'][t] = baseline[t]

    return feasible_flex


def prepare_flex(agent_flexibilities, logger):
    """
    Prepares flexibility data from protocol buffer messages for optimization.

    This function processes flexibility data from multiple agents and time intervals,
    corrects any inconsistencies, and organizes the data by start timestamp.

    Args:
        agent_flexibilities (dict): Dictionary mapping agent IDs to their flexibility data
            in protocol buffer format.
        logger: Logger object for logging warnings and errors.

    Returns:
        dict: Dictionary organized by start timestamp, containing processed and validated
              flexibility data for all agents.
    """
    feasible_flex = {}

    for agent, data in agent_flexibilities.items():
        # Extract baseline start time and values
        bl_start = data.t_start
        baseline = data.baseline_values

        # Convert protocol buffer flexibility values to dictionary
        flex_dict = convert_proto_flex(data.flex_values)

        # Process each flexibility time interval
        for t_start, flex in flex_dict.items():
            # Initialize data structure for this timestamp if not exists
            if t_start not in feasible_flex.keys():
                feasible_flex[t_start] = {
                    'aggre_bl': []}

            # Calculate offset between flexibility start and baseline start (in 15-min intervals)
            bl_offset = int((t_start - bl_start) / 900)

            # Check for and fix inconsistencies between flexibility and baseline
            for t in range(len(flex['minP'])):
                # Fix minP if it's greater than baseline
                if flex['minP'][t] > baseline[t + bl_offset]:
                    flex['minP'][t] = baseline[t + bl_offset]
                    # Log significant discrepancies (more than rounding errors)
                    if abs(flex['minP'][t] - baseline[t + bl_offset]) > 1:
                        logger.warn(f"{agent}: flex {flex['minP'][t]} and BL {baseline[t + bl_offset]} not compatible")

                # Fix maxP if it's less than baseline
                if flex['maxP'][t] < baseline[t + bl_offset]:
                    flex['maxP'][t] = baseline[t + bl_offset]
                    # Log significant discrepancies
                    if abs(flex['maxP'][t] - baseline[t + bl_offset]) > 1:
                        logger.warn(f"{agent}: flex {flex['maxP'][t]} and BL {baseline[t + bl_offset]} not compatible")

            # Store the processed data in the result dictionary
            feasible_flex[t_start][agent] = {
                't_start': t_start,
                'baseline_values': baseline[bl_offset:],
                'costs_ch': data.costs_ch,
                'costs_dis': data.costs_dis,
                'flex_values': flex
            }

            # Update the aggregated baseline for this timestamp
            feasible_flex[t_start]['aggre_bl'] = [
                feasible_flex[t_start]['aggre_bl'][i] + baseline[bl_offset:][i] if len(
                    feasible_flex[t_start]['aggre_bl']) > i else 0 + baseline[bl_offset:][i] for i in
                range(len(baseline[bl_offset:]))]

    return feasible_flex


def aggregate_baselines(agent_flexibilities):
    """
    Aggregates baseline values from multiple agents organized by timestamp.

    This function collects all unique start timestamps from the agent data
    and calculates the sum of baseline values for each timestamp.

    Args:
        agent_flexibilities (dict): Dictionary mapping agent IDs to their flexibility data
            in protocol buffer format.

    Returns:
        dict: Dictionary mapping start timestamps to lists of aggregated baseline values.
    """
    # Initialize dictionary to store aggregated baselines
    aggr_baseline = {}

    # Find all unique start timestamps across all agents
    unique_t_start = set([a_flex.t_start for a_flex in agent_flexibilities.values()])

    # Process each agent's data
    for agent, data in agent_flexibilities.items():
        bl_start = data.t_start
        baseline = data.baseline_values

        # Initialize entry for this baseline start time if not exists
        if bl_start not in aggr_baseline.keys():
            aggr_baseline[bl_start] = []

        # Aggregate baseline values for each unique timestamp
        for t_start in unique_t_start:
            # Calculate offset between this timestamp and baseline start
            bl_offset = int((t_start - bl_start) / 900)

            # Add baseline values to the aggregated values list
            aggr_baseline[t_start] = [
                aggr_baseline[t_start][i] + baseline[bl_offset:][i] if len(
                    aggr_baseline[t_start]) > i else 0 + baseline[bl_offset:][i] for i in
                range(len(baseline[bl_offset:]))]

    return aggr_baseline


def get_flex_for_t_start(flex, t_start):
    """
    Extracts flexibility data for a specific start timestamp.

    This function filters the flexibility data to include only the data
    for the specified timestamp.

    Args:
        flex (dict): Dictionary mapping agent IDs to their flexibility data.
        t_start (int): Start timestamp to filter for.

    Returns:
        dict: Dictionary containing flexibility data for the specified timestamp.
    """
    flex_for_t = {}

    for agent, data in flex.items():
        baseline = data.baseline_values

        # Store the data in the result dictionary
        flex_for_t[agent] = {
            't_start': data.t_start,
            'baseline_values': baseline,
            'costs_ch': data.costs_ch,
            'costs_dis': data.costs_dis,
            'flex_values': data
        }

    return flex_for_t


def convert_proto_flex(flex_values):
    """
    Converts protocol buffer flexibility values to a dictionary format.

    This function processes flexibility values from protocol buffer format
    into a nested dictionary organized by start timestamp and flexibility type.

    Args:
        flex_values (list): List of flexibility values in protocol buffer format.

    Returns:
        dict: Dictionary mapping start timestamps to dictionaries of flexibility types ('minP', 'maxP')
              and their corresponding values.
    """
    flex_dict = {}

    # Process each flexibility value
    for f in flex_values:
        # Convert flexibility type code to string
        f_type = f.flexibility_type
        if f_type == 0:
            flex_type = 'minP'
        else:
            flex_type = 'maxP'

        # Initialize entry for this start time if not exists
        if f.t_start not in flex_dict.keys():
            flex_dict[f.t_start] = {}

        # Store the flexibility values
        flex_dict[f.t_start][flex_type] = f.value

    # Return sorted dictionary by start time
    return dict(sorted(flex_dict.items()))