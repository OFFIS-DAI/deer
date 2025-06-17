import datetime
import math
import random
import pandas as pd
import time
import os

from logic.src.aggregation_logic import nl_optimize_bi
from logic.src.baseline_logic import optimize_household_baseline
from logic.src.flexibility_logic import get_battery_flexibility_bands
from simulation.src.market_restricted_flexibility_calculation import get_valid_battery_flexibility
from simulation.src.assets import SimulatedBatteryStorage


def run_simulation_and_save_results():
    """
    Run baseline optimization, calculate flexibility, perform aggregation,
    and save results for analysis.
    """
    random.seed = 3

    NUM_INTERVALS = 3 * 24 * 4  # three days
    START_TIME = datetime.datetime(2024, 7, 12, 6, 0)

    """
    pre-configure input data
    """
    # prices
    j_cons_constant = [0.1] * NUM_INTERVALS  # constant prices for power consumption
    j_cons_with_incentive = [random.choice([0, 0.1, 0.2]) for _ in
                             range(NUM_INTERVALS)]  # prices for incentive for charging

    j_feed_in = [-0.0086] * NUM_INTERVALS  # prices for feed-in

    def empty_battery():
        return 0

    def full_battery():
        return 1

    def random_battery():
        return random.random()

    def normal_dist_battery():
        return max(0, min(1, random.normalvariate(0.5, 0.25)))

    # fixed power (from household and PV)
    def p_fixed_equal_cons_feed_in():
        """
        Consumption == feed-in in all intervals.
        Returns a list of zeros.
        """
        return [0] * NUM_INTERVALS

    def p_fixed_higher_cons_in_one_interval():
        """
        Higher consumption in one interval.
        """
        return [500] + [0] * (NUM_INTERVALS - 1)

    def p_fixed_higher_cons_random_interval():
        """
        Higher consumption in one random interval.
        """
        rand_int = random.randint(0, NUM_INTERVALS - 1)
        return [0 if i != rand_int else 500 for i in range(NUM_INTERVALS)]

    def p_fixed_higher_cons_or_feed_in():
        """
        Higher consumption or feed-in in the tenth interval.
        """
        return [random.choice([500, -500]) if i == 10 else 0 for i in range(NUM_INTERVALS)]

    def p_fixed_random():
        """
        Totally random fixed power values.
        """
        return [random.random() * random.choice([-500, 500]) for _ in range(NUM_INTERVALS)]

    small_bat = SimulatedBatteryStorage(asset_id='bat_small',
                                        f_start=0,
                                        p_start=0,
                                        capacity=6500,
                                        max_power_w=5000,
                                        min_power_w=-5000,
                                        efficiency_charge=0.95,
                                        efficiency_discharge=0.95,
                                        investment_costs_euro=2031,
                                        number_of_full_cycles=6000)
    medium_bat = SimulatedBatteryStorage(asset_id='bat_medium',
                                         f_start=0.5,
                                         p_start=0,
                                         capacity=9000,
                                         max_power_w=10200,
                                         min_power_w=-10200,
                                         efficiency_charge=0.95,
                                         efficiency_discharge=0.95,
                                         investment_costs_euro=3635,
                                         number_of_full_cycles=6000)

    large_bat = SimulatedBatteryStorage(asset_id='bat_large',
                                        f_start=0.5,
                                        p_start=0,
                                        capacity=12000,
                                        max_power_w=12300,
                                        min_power_w=-12300,
                                        efficiency_charge=0.95,
                                        efficiency_discharge=0.95,
                                        investment_costs_euro=5909,
                                        number_of_full_cycles=6000)

    # List of available batteries
    batteries = [small_bat, medium_bat, large_bat]

    """
    Configure runs
    """

    input_sets = {
        'number_of_assets': [10, 100, 1000],
        'prices': [('incentive', j_cons_with_incentive),
                   ('constant', j_cons_constant)
                   ],
        'fixed_power_generators': [
            ('random fixed power', p_fixed_random),
            ('consumption equal to feed-in', p_fixed_equal_cons_feed_in),
            ('higher consumption in one interval', p_fixed_higher_cons_in_one_interval),
            ('higher consumption or feed-in', p_fixed_higher_cons_or_feed_in),
            ('higher consumption in random interval', p_fixed_higher_cons_random_interval)
        ],
        'start_soc': [
            ('all normal distributed', normal_dist_battery),
            ('all empty', empty_battery),
            ('all full', full_battery),
            ('all random', random_battery)
        ]
    }

    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to find the repo root folder "deer"
    # This assumes the current script is somewhere within the deer repository
    repo_root = script_dir
    while os.path.basename(repo_root) != "deer" and os.path.dirname(repo_root) != repo_root:
        repo_root = os.path.dirname(repo_root)

    # If we couldn't find "deer" directory in the path, default to current directory
    if os.path.basename(repo_root) != "deer":
        print("Warning: Could not locate 'deer' repository root. Using current directory.")
        repo_root = script_dir

    # Define the results directory relative to the repo root
    results_dir = os.path.join(repo_root, "simulation", "simulation_studies", "flexibility_comparison",
                               "aggregation_results")
    os.makedirs(results_dir, exist_ok=True)

    # Create a DataFrame to store aggregation results
    results_data = []

    # Create a list to store calculation times
    calculation_time_data = []

    baseline_data = []

    def run_calculations():
        # Store the data in the result dictionary
        flexibilities_opt = [{} for _ in range(NUM_INTERVALS - 4)]
        flexibilities_sum = [{} for _ in range(NUM_INTERVALS - 4)]

        # Track calculation times for this scenario
        baseline_calc_times = []
        flex_calc_times = []

        for asset_idx in range(n_assets):
            battery = batteries[asset_idx % len(batteries)]
            start_soc = start_soc_generator()
            battery.reset_f_start(start_soc)
            battery_type = battery.asset_id

            # Generate asset-specific fixed power values
            # Call the function to generate new values for each asset
            fixed_power = f_power_generator()

            # Measure baseline calculation time
            baseline_start_time = time.time()

            # Run the optimization
            p_bat, p_grid_to_ev, p_bat_to_ev, p_hp, total_cost, soc_values_bat, soc_values_hp, non_fulfilled_fixed_power_values = (
                optimize_household_baseline(
                    num_intervals=NUM_INTERVALS,
                    start_timestamp=START_TIME.timestamp(),
                    prices_consumption_cent_per_wh=price,
                    prices_feed_in_cent_per_wh=j_feed_in,
                    fixed_power=fixed_power,  # Use the asset-specific fixed power values
                    heat_demand=None,
                    ambient_temperature=None,
                    heat_pump=None,
                    battery_storage=battery,
                    assumed_fixed_power_values=[],
                    batch_size=NUM_INTERVALS
                ))

            baseline_calc_time = time.time() - baseline_start_time
            baseline_calc_times.append(baseline_calc_time)

            # Store calculation time data for each asset
            calculation_time_data.append({
                'scenario': scenario_key,
                'asset_idx': asset_idx,
                'battery_type': battery_type,
                'calculation_type': 'baseline',
                'calculation_time': baseline_calc_time
            })
            if n_assets < 1000:
                baseline_data.append({
                    'scenario': scenario_key,
                    'asset_idx': asset_idx,
                    'battery_type': battery_type,
                    'baseline_values': p_bat,
                    'SoC_values': soc_values_bat
                })

            # Measure flexibility calculation time
            flex_start_time = time.time()

            # Calculate market-restricted flexibility
            sum_flexibility = get_valid_battery_flexibility(battery_storage=battery,
                                                            num_intervals=NUM_INTERVALS - 4,
                                                            provisioning_intervals=4,
                                                            baseline_power=p_bat,
                                                            # only one provisioning interval for this type
                                                            baseline_soc=soc_values_bat)

            flex_bands = get_battery_flexibility_bands(battery_storage=battery,
                                                       num_intervals=NUM_INTERVALS - 4,
                                                       provisioning_intervals=4,
                                                       baseline_power=p_bat,
                                                       # only one provisioning interval for this type
                                                       baseline_soc=soc_values_bat)

            flex_calc_time = time.time() - flex_start_time
            flex_calc_times.append(flex_calc_time)

            for i, flex in enumerate(flex_bands):
                for j in range(4):
                    if p_bat[i + j] < 0:
                        # charge
                        if not flex['minP'][j] <= p_bat[i + j]:
                            print('_' * 50)
                            print('interval: ', j)
                            print('minP: ', flex['minP'])
                            print('p_bat: ', p_bat[i + j])
                    if p_bat[i + j] > 0:
                        # discharge
                        if not flex['maxP'][j] >= p_bat[i + j]:
                            print('_' * 50)
                            print('interval: ', j)
                            print('maxP: ', flex['maxP'])
                            print('p_bat: ', p_bat[i + j])

            # Store calculation time data for flexibility calculation
            calculation_time_data.append({
                'scenario': scenario_key,
                'asset_idx': asset_idx,
                'battery_type': battery_type,
                'calculation_type': 'flexibility',
                'calculation_time': flex_calc_time
            })

            for i in range(NUM_INTERVALS - 4):
                # Store the data in the result dictionary
                flexibilities_opt[i][f'asset{asset_idx}'] = {
                    'baseline_values': p_bat[i:i+4],
                    'costs_ch': 0,
                    'costs_dis': 0,
                    'flex_values': flex_bands[i]
                }
                # Ensure flexibility limits are consistent with baseline
                for t in range(len(flex_bands[i]['minP'])):
                    # If minP is greater than baseline, set it equal to baseline
                    if flex_bands[i]['minP'][t] > p_bat[t]:
                        flexibilities_opt[i][f'asset{asset_idx}']["flex_values"]['minP'][t] = p_bat[t]

                    # If maxP is less than baseline, set it equal to baseline
                    if flex_bands[i]['maxP'][t] < p_bat[t]:
                        flexibilities_opt[i][f'asset{asset_idx}']["flex_values"]['maxP'][t] = p_bat[t]

                flexibilities_sum[i][f'asset{asset_idx}'] = sum_flexibility[i]

        total_aggregation_time = 0
        # For each time step, perform aggregation and save results
        for i in range(NUM_INTERVALS - 4):
            # Measure aggregation calculation time
            aggregation_start_time = time.time()

            # Optimize min flexibility
            opt_min_flexibility, _, _ = nl_optimize_bi(
                flexibilities_opt[i],
                4,
                virtual_lb=False,
                flex_type='min'
            )

            min_aggregation_time = time.time() - aggregation_start_time
            total_aggregation_time += min_aggregation_time

            # Reset timer for max flex calculation
            aggregation_start_time = time.time()

            # Optimize max flexibility
            opt_max_flexibility, _, _ = nl_optimize_bi(
                flexibilities_opt[i],
                4,
                virtual_lb=False,
                flex_type='max'
            )

            max_aggregation_time = time.time() - aggregation_start_time
            total_aggregation_time += max_aggregation_time

            # Calculate sum of min and max flexibility
            sum_min_flexibility = sum([flex['minP'] for flex in flexibilities_sum[i].values()])
            sum_max_flexibility = sum([flex['maxP'] for flex in flexibilities_sum[i].values()])

            # Store results in the DataFrame
            results_data.append({
                'scenario': scenario_key,
                'interval': i,
                'opt_min_flexibility': math.ceil(opt_min_flexibility),
                'opt_max_flexibility': math.floor(opt_max_flexibility),
                'sum_min_flexibility': math.ceil(sum_min_flexibility),
                'sum_max_flexibility': math.floor(sum_max_flexibility)
            })

        # Store calculation time data for aggregation
        calculation_time_data.append({
            'scenario': scenario_key,
            'battery_type': 'all',  # Aggregation involves all batteries
            'calculation_type': 'aggregation_opt',
            'calculation_time': total_aggregation_time,
            'num_assets': n_assets
        })

    """
    Run baseline optimization and aggregation and save results
    """
    for n_assets in input_sets['number_of_assets']:
        for n_price, price in input_sets['prices']:
            for n_f_power, f_power_generator in input_sets['fixed_power_generators']:
                for soc_config_n, start_soc_generator in input_sets['start_soc']:
                    scenario_key = f"{n_price} - {n_f_power} - {n_assets} assets - {soc_config_n}"
                    print('Run baseline optimization for scenario with key: ', scenario_key)
                    run_calculations()

    for n_assets in [1000, 10000]:
        n_price, price = input_sets['prices'][0]
        n_f_power, f_power_generator = input_sets['fixed_power_generators'][0]
        soc_config_n, start_soc_generator = input_sets['start_soc'][0]
        scenario_key = (f"{n_price} - {n_f_power} - {n_assets} assets - "
                        f"{soc_config_n}")
        print('Run baseline optimization for scenario with key: ', scenario_key)
        run_calculations()

    # Create a DataFrame from the results and save to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f"{results_dir}/aggregation_results.csv", index=False)

    # Create a DataFrame from the calculation times and save to CSV
    calculation_times_df = pd.DataFrame(calculation_time_data)
    calculation_times_df.to_csv(f"{results_dir}/calculation_times.csv", index=False)

    # Create a DataFrame from the baseline opt results and save to CSV
    baseline_df = pd.DataFrame(baseline_data)
    baseline_df.to_csv(f"{results_dir}/baseline_data.csv", index=False)

    return results_df, calculation_times_df


if __name__ == "__main__":
    results_df, calculation_times_df = run_simulation_and_save_results()
    print("Simulation complete. Results saved to 'aggregation_results' directory.")
    print(f"Aggregation results saved to 'aggregation_results/aggregation_results.csv'")
    print(f"Calculation times saved to 'aggregation_results/calculation_times.csv'")
    print(f"Baseline results saved to 'aggregation_results/baseline_data.csv'")