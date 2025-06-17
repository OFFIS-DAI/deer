import math

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import plot
import os
pio.kaleido.scope.mathjax = None

# Set default theme for better paper appearance
pio.templates.default = "plotly_white"

# Define a better color palette (more publication-friendly)
# Using a colorblind-friendly palette
COLOR_PALETTE = {
    'optimized': '#0072B2',  # Blue
    'market_restricted': '#E69F00',  # Orange
    'asset_10': '#009E73',  # Green
    'asset_100': '#D55E00',  # Red-orange
    'asset_1000': '#AA5042',
    'asset_10000': '#CC79A7'  # Pink/magenta
}


def load_and_process_data():
    """Load and preprocess the data for plotting."""
    print("Loading data files from 'aggregation_results' folder...")

    # Define file paths
    aggregation_results_path = os.path.join("aggregation_results", "aggregation_results.csv")
    calculation_times_path = os.path.join("aggregation_results", "calculation_times.csv")

    # Load the aggregation results
    aggregation_results = pd.read_csv(aggregation_results_path)

    # Load calculation times
    calculation_times = pd.read_csv(calculation_times_path)

    # Process the data
    def extract_num_assets(scenario):
        """Extract the number of assets from the scenario string."""
        if not isinstance(scenario, str):
            return None
        return int(scenario.split(" - ")[-2].split()[0])

    def extract_scenario_type(scenario):
        """Extract the scenario type (constant or incentive)."""
        if not isinstance(scenario, str):
            return None
        return scenario.split(" - ")[0]

    def extract_flex_pattern(scenario):
        """Extract the flexibility pattern from the scenario string."""
        if not isinstance(scenario, str):
            return None
        return scenario.split(" - ")[1]

    # Add columns with extracted information
    aggregation_results['num_assets'] = aggregation_results['scenario'].apply(extract_num_assets)
    aggregation_results['scenario_type'] = aggregation_results['scenario'].apply(extract_scenario_type)
    aggregation_results['flex_pattern'] = aggregation_results['scenario'].apply(extract_flex_pattern)

    calculation_times['num_assets'] = calculation_times['scenario'].apply(extract_num_assets)
    calculation_times['scenario_type'] = calculation_times['scenario'].apply(extract_scenario_type)

    # Calculate total flexibility range
    aggregation_results['opt_flex_range'] = aggregation_results['opt_max_flexibility'] - aggregation_results[
        'opt_min_flexibility']
    aggregation_results['sum_flex_range'] = aggregation_results['sum_max_flexibility'] - aggregation_results[
        'sum_min_flexibility']

    return aggregation_results, calculation_times


def generate_calculation_time_table(calculation_times):
    """Generate a LaTeX table of calculation times."""
    print("Generating LaTeX table for calculation times...")

    # Filter relevant calculation types
    calc_types = ['baseline', 'flexibility', 'aggregation_opt']

    # Prepare data for the table
    calculation_summary = calculation_times[
        calculation_times['calculation_type'].isin(calc_types)
    ].groupby(['num_assets', 'calculation_type']).agg({
        'calculation_time': 'mean'
    }).reset_index()

    # Pivot the data to create a table with calculation types as columns
    # and number of assets as rows
    table_data = calculation_summary.pivot(
        index='num_assets',
        columns='calculation_type',
        values='calculation_time'
    ).reset_index()

    # Sort by number of assets
    table_data = table_data.sort_values('num_assets')

    # Begin generating LaTeX table
    latex_table = []
    latex_table.append("\\begin{table}[htbp]")
    latex_table.append("\\centering")
    latex_table.append("\\caption{Average Calculation Times for Different Components (in seconds)}")
    latex_table.append("\\label{tab:calculation-times}")

    # Table header
    latex_table.append("\\begin{tabular}{p{3.5cm}|ccc}")
    latex_table.append("\\toprule")
    latex_table.append(
        "\\textbf{Number of Assets} & \\textbf{Baseline} & \\textbf{Flexibility} & \\textbf{Aggregation} \\\\")
    latex_table.append("\\midrule")

    # Table rows
    for _, row in table_data.iterrows():
        num_assets = int(row['num_assets'])
        baseline_time = row.get('baseline', float('nan'))
        flexibility_time = row.get('flexibility', float('nan'))
        aggregation_time = row.get('aggregation_opt', float('nan'))

        # Format the times with 3 decimal places
        baseline_formatted = f"{baseline_time:.3f}" if not np.isnan(baseline_time) else "N/A"
        flexibility_formatted = f"{flexibility_time:.3f}" if not np.isnan(flexibility_time) else "N/A"
        aggregation_formatted = f"{aggregation_time:.1f}" if not np.isnan(aggregation_time) else "N/A"

        latex_table.append(
            f"{num_assets} & {baseline_formatted} & {flexibility_formatted} & {aggregation_formatted} \\\\")

    # Table footer
    latex_table.append("\\bottomrule")
    latex_table.append("\\end{tabular}")
    latex_table.append("\\end{table}")

    # Join all lines into a single string and return
    return "\n".join(latex_table)

def plot_flexibility_comparison(aggregation_results):
    """
    Create a compact, publication-friendly plot comparing flexibility between
    optimized and sum approaches with three subplots for pricing, SoC configurations,
    and power configurations.
    """
    print("Creating compact flexibility comparison plot...")

    # Filter for 100 assets only
    aggregation_results_100 = aggregation_results#[aggregation_results['num_assets'] == 100]

    # Add flexibility range columns if they don't exist
    if 'opt_flex_range' not in aggregation_results_100.columns:
        aggregation_results_100['opt_flex_range'] = aggregation_results_100['opt_max_flexibility'] - \
                                                    aggregation_results_100['opt_min_flexibility']
    if 'sum_flex_range' not in aggregation_results_100.columns:
        aggregation_results_100['sum_flex_range'] = aggregation_results_100['sum_max_flexibility'] - \
                                                    aggregation_results_100['sum_min_flexibility']

    # Create a subplot with three panels in a 1x3 grid (horizontal layout)
    fig = make_subplots(
        rows=1, cols=3,

        horizontal_spacing=0.04,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )

    # ======== SUBPLOT 1: Pricing Comparison ========
    pricing_summary = aggregation_results_100.groupby(['scenario_type']).agg({
        'opt_flex_range': 'mean',
        'sum_flex_range': 'mean'
    }).reset_index()

    # Better pricing scenario labels
    pricing_summary['pricing_label'] = pricing_summary['scenario_type'].apply(
        lambda x: 'Constant' if x == 'constant' else 'Incentive'
    )

    # Sort data for consistent ordering
    pricing_summary = pricing_summary.sort_values('pricing_label')

    # Add traces for pricing scenarios
    fig.add_trace(
        go.Bar(
            x=pricing_summary['pricing_label'],
            y=pricing_summary['opt_flex_range'],
            name='Optimized',
            marker_color=COLOR_PALETTE['optimized'],
            offsetgroup=0
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=pricing_summary['pricing_label'],
            y=pricing_summary['sum_flex_range'],
            name='Market-Restricted',
            marker_color=COLOR_PALETTE['market_restricted'],
            offsetgroup=1
        ),
        row=1, col=1
    )

    # Add percentage improvements for pricing (smaller font)
    for i, row in pricing_summary.iterrows():
        if row['sum_flex_range'] == 0:
            improvement = math.inf
        else:
            improvement = ((row['opt_flex_range'] - row['sum_flex_range']) / row['sum_flex_range']) * 100
        if improvement > 0:
            fig.add_annotation(
                x=row['pricing_label'],
                y=max(row['opt_flex_range'], row['sum_flex_range']) * 1.05,
                text=f"+{improvement:.1f}%",
                showarrow=False,
                font=dict(size=9, color="#333333", family="Arial"),
                row=1, col=1
            )

    # ======== SUBPLOT 2: SoC Configuration Comparison ========
    # Extract SoC config from scenario string
    def extract_soc_config(scenario):
        """Extract the SoC configuration from the scenario string."""
        if not isinstance(scenario, str):
            return None
        parts = scenario.split(" - ")
        if len(parts) >= 4:
            return parts[3]  # The SoC config is the 4th part
        return None

    aggregation_results_100['soc_config'] = aggregation_results_100['scenario'].apply(extract_soc_config)

    soc_summary = aggregation_results_100.groupby(['soc_config']).agg({
        'opt_flex_range': 'mean',
        'sum_flex_range': 'mean'
    }).reset_index()

    # Better SoC labels
    def clean_soc_label(label):
        if label == 'all normal distributed':
            return 'Normal'
        elif label == 'all empty':
            return 'Empty'
        elif label == 'all full':
            return 'Full'
        elif label == 'all random':
            return 'Random'
        return label

    soc_summary['soc_label'] = soc_summary['soc_config'].apply(clean_soc_label)

    # Sort data for consistent ordering
    soc_summary = soc_summary.sort_values('soc_label')

    # Add traces for SoC scenarios
    fig.add_trace(
        go.Bar(
            x=soc_summary['soc_label'],
            y=soc_summary['opt_flex_range'],
            name='Optimized',
            marker_color=COLOR_PALETTE['optimized'],
            offsetgroup=0,
            showlegend=False
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(
            x=soc_summary['soc_label'],
            y=soc_summary['sum_flex_range'],
            name='Market-Restricted',
            marker_color=COLOR_PALETTE['market_restricted'],
            offsetgroup=1,
            showlegend=False
        ),
        row=1, col=2
    )

    # Add percentage improvements for SoC (smaller font)
    for i, row in soc_summary.iterrows():
        if row['sum_flex_range'] == 0:
            improvement = math.inf
        else:
            improvement = ((row['opt_flex_range'] - row['sum_flex_range']) / row['sum_flex_range']) * 100
        if improvement > 0:
            fig.add_annotation(
                x=row['soc_label'],
                y=max(row['opt_flex_range'], row['sum_flex_range']) * 1.05,
                text=f"+{improvement:.1f}%",
                showarrow=False,
                font=dict(size=9, color="#333333", family="Arial"),
                row=1, col=2
            )

    # ======== SUBPLOT 3: Power Configuration Comparison ========
    # Extract flexibility pattern from scenario string
    def extract_flex_pattern(scenario):
        """Extract the flexibility pattern from the scenario string."""
        if not isinstance(scenario, str):
            return None
        parts = scenario.split(" - ")
        if len(parts) >= 2:
            return parts[1]  # The flex pattern is the 2nd part
        return None

    aggregation_results_100['flex_pattern'] = aggregation_results_100['scenario'].apply(extract_flex_pattern)

    power_summary = aggregation_results_100.groupby(['flex_pattern']).agg({
        'opt_flex_range': 'mean',
        'sum_flex_range': 'mean'
    }).reset_index()

    # Simplify pattern names for better readability
    def simplify_pattern(pattern):
        if pattern == 'consumption equal to feed-in':
            return 'Equal cons/feed-in'
        elif pattern == 'higher consumption in one interval':
            return 'One Peak'
        elif pattern == 'higher consumption in random interval':
            return ('Random Peak')
        elif pattern == 'higher consumption or feed-in':
            return 'Higher cons/feed-in'
        elif pattern == 'random fixed power':
            return 'Random'
        return pattern

    power_summary['pattern_label'] = power_summary['flex_pattern'].apply(simplify_pattern)

    # Sort data by pattern name
    power_summary = power_summary.sort_values('pattern_label')

    # Add traces for power config scenarios
    fig.add_trace(
        go.Bar(
            x=power_summary['pattern_label'],
            y=power_summary['opt_flex_range'],
            name='Optimized',
            marker_color=COLOR_PALETTE['optimized'],
            offsetgroup=0,
            showlegend=False
        ),
        row=1, col=3
    )

    fig.add_trace(
        go.Bar(
            x=power_summary['pattern_label'],
            y=power_summary['sum_flex_range'],
            name='Market-Restricted',
            marker_color=COLOR_PALETTE['market_restricted'],
            offsetgroup=1,
            showlegend=False
        ),
        row=1, col=3
    )

    # Add percentage improvements for power configs (smaller font)
    for i, row in power_summary.iterrows():
        if row['sum_flex_range'] == 0:
            improvement = math.inf
        else:
            improvement = ((row['opt_flex_range'] - row['sum_flex_range']) / row['sum_flex_range']) * 100
        if improvement > 0:
            fig.add_annotation(
                x=row['pattern_label'],
                y=max(row['opt_flex_range'], row['sum_flex_range']) * 1.05,
                text=f"+{improvement:.1f}%",
                showarrow=False,
                font=dict(size=9, color="#333333", family="Arial"),
                row=1, col=3
            )

    # Update layout for a more compact figure
    fig.update_layout(
        title=None,  # Remove title to save space
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        height=250,  # Reduced height for compact display
        width=800,  # Reduced width
        margin=dict(t=60, b=50, l=50, r=30),  # Reduced margins
        font=dict(size=10),  # Smaller font size
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)'
    )

    # Smaller tick font size on all axes
    fig.update_xaxes(tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))

    # Format y-axis labels for large numbers
    fig.update_yaxes(
        title=None,  # Remove y-axis titles to save space
        tickprefix="",
        ticksuffix="",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)'
    )

    # Only show y-axis title on the first subplot
    fig.update_yaxes(title="Flexibility Range (W)", row=1, col=1)

    # Create directory for output if it doesn't exist
    output_dir = "paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    pio.write_image(fig, os.path.join(output_dir, 'flexibility_comparison.pdf'))
    pio.write_image(fig, os.path.join(output_dir, 'flexibility_comparison.png'), scale=3, width=800, height=400)
    plot(fig, filename=os.path.join(output_dir, 'flexibility_comparison.html'), auto_open=False)
    print(f"Compact flexibility comparison plot saved to {output_dir} folder.")

    return fig

def plot_soc_flexibility(aggregation_results):
    """Create a plot showing min and max flexibility for different SoC configurations,
    combining both pricing scenarios in a single plot."""
    print("Creating SoC min/max flexibility comparison plot...")

    # Define function to extract SoC config from scenario string
    def extract_soc_config(scenario):
        """Extract the SoC configuration from the scenario string."""
        if not isinstance(scenario, str):
            return None
        parts = scenario.split(" - ")
        if len(parts) >= 4:
            return parts[3]  # The SoC config is the 4th part
        return None

    # Add SoC configuration column
    aggregation_results['soc_config'] = aggregation_results['scenario'].apply(extract_soc_config)

    # Group data by SoC config and number of assets (combining pricing scenarios)
    soc_flexibility_summary = aggregation_results.groupby(
        ['soc_config', 'num_assets']
    ).agg({
        'opt_min_flexibility': 'mean',  # Aggregated min flexibility values
        'opt_max_flexibility': 'mean',  # Aggregated max flexibility values
        'sum_min_flexibility': 'mean',  # Sum of individual min flexibility values
        'sum_max_flexibility': 'mean'  # Sum of individual max flexibility values
    }).reset_index()

    # Focus on scenarios with 100 assets for cleaner visualization
    plot_data = soc_flexibility_summary#[soc_flexibility_summary['num_assets'] == 100]

    # Create two subplots: one for min flexibility, one for max flexibility
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Minimum Flexibility by Initial SoC", "Maximum Flexibility by Initial SoC"),
        vertical_spacing=0.15
    )

    # Colors for optimized and market-restricted approaches
    optimized_color = COLOR_PALETTE['optimized']
    market_color = COLOR_PALETTE['market_restricted']

    # Add traces for MIN flexibility
    fig.add_trace(
        go.Bar(
            x=plot_data['soc_config'],
            y=plot_data['opt_min_flexibility'],
            name='Optimized Approach',
            marker_color=optimized_color,
            offsetgroup=0
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=plot_data['soc_config'],
            y=plot_data['sum_min_flexibility'],
            name='Market-Restricted Approach',
            marker_color=market_color,
            offsetgroup=1
        ),
        row=1, col=1
    )

    # Add traces for MAX flexibility
    fig.add_trace(
        go.Bar(
            x=plot_data['soc_config'],
            y=plot_data['opt_max_flexibility'],
            name='Optimized Approach',
            marker_color=optimized_color,
            offsetgroup=0,
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=plot_data['soc_config'],
            y=plot_data['sum_max_flexibility'],
            name='Market-Restricted Approach',
            marker_color=market_color,
            offsetgroup=1,
            showlegend=False
        ),
        row=2, col=1
    )

    # Add annotations for percentage differences for MIN flexibility
    for i, row in plot_data.iterrows():
        if row['sum_min_flexibility'] == 0:
            continue
        difference = ((row['opt_min_flexibility'] - row['sum_min_flexibility']) / abs(row['sum_min_flexibility'])) * 100
        if abs(difference) > 5:  # Only show significant differences
            fig.add_annotation(
                x=row['soc_config'],
                y=min(row['opt_min_flexibility'], row['sum_min_flexibility']) * 0.9,
                text=f"{difference:.1f}%",
                showarrow=False,
                font=dict(size=10, color="#333333", family="Arial"),
                row=1, col=1
            )

    # Add annotations for percentage differences for MAX flexibility
    for i, row in plot_data.iterrows():
        if row['sum_max_flexibility'] == 0:
            continue
        difference = ((row['opt_max_flexibility'] - row['sum_max_flexibility']) / abs(row['sum_max_flexibility'])) * 100
        if abs(difference) > 5:  # Only show significant differences
            fig.add_annotation(
                x=row['soc_config'],
                y=max(row['opt_max_flexibility'], row['sum_max_flexibility']) * 1.05,
                text=f"{difference:.1f}%",
                showarrow=False,
                font=dict(size=10, color="#333333", family="Arial"),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(
        title_text="Min and Max Flexibility by Initial Battery State of Charge",
        title_font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        height=800,
        width=1000,
        margin=dict(t=150, b=100, l=80, r=40),
        font=dict(size=14),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)'
    )

    # Format x-axis for better readability
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=12)
    )

    # Format y-axis for each subplot
    fig.update_yaxes(
        title="Minimum Flexibility (W)",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)',
        row=1, col=1
    )

    fig.update_yaxes(
        title="Maximum Flexibility (W)",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)',
        row=2, col=1
    )

    # Create directory for output if it doesn't exist
    output_dir = "paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    pio.write_image(fig, os.path.join(output_dir, 'soc_min_max_flexibility.pdf'))
    pio.write_image(fig, os.path.join(output_dir, 'soc_min_max_flexibility.png'), scale=3)
    plot(fig, filename=os.path.join(output_dir, 'soc_min_max_flexibility.html'), auto_open=False)
    print(f"SoC min/max flexibility comparison plot saved to {output_dir} folder.")

    return fig


def plot_calculation_times(calculation_times):
    """Create a plot comparing calculation times for different components."""
    print("Creating calculation time plot...")

    # Filter relevant calculation types
    calc_types = ['baseline', 'flexibility', 'aggregation_opt']

    # Function to clean calculation type labels
    def clean_calc_type(calc_type):
        if calc_type == 'aggregation_opt':
            return 'Aggregation'
        else:
            return calc_type.capitalize()

    # Prepare data for plotting
    calculation_summary = calculation_times[
        calculation_times['calculation_type'].isin(calc_types)
    ].groupby(['num_assets', 'calculation_type']).agg({
        'calculation_time': 'mean'
    }).reset_index()

    # Apply the clean function to calculation type
    calculation_summary['calculation_type_clean'] = calculation_summary['calculation_type'].apply(clean_calc_type)

    # Create figure
    fig = go.Figure()

    # Filter data for 10, 100, and 1000 assets
    data_10 = calculation_summary[calculation_summary['num_assets'] == 10]
    data_100 = calculation_summary[calculation_summary['num_assets'] == 100]
    data_1000 = calculation_summary[calculation_summary['num_assets'] == 1000]
    data_10000 = calculation_summary[calculation_summary['num_assets'] == 10000]

    # Sort by calculation type for consistent ordering
    ordered_calc_types = ['Baseline', 'Flexibility', 'Aggregation']
    data_10['order'] = data_10['calculation_type_clean'].apply(
        lambda x: ordered_calc_types.index(x) if x in ordered_calc_types else 999)
    data_100['order'] = data_100['calculation_type_clean'].apply(
        lambda x: ordered_calc_types.index(x) if x in ordered_calc_types else 999)
    data_1000['order'] = data_1000['calculation_type_clean'].apply(
        lambda x: ordered_calc_types.index(x) if x in ordered_calc_types else 999)
    data_10000['order'] = data_10000['calculation_type_clean'].apply(
        lambda x: ordered_calc_types.index(x) if x in ordered_calc_types else 999)

    data_10 = data_10.sort_values('order')
    data_100 = data_100.sort_values('order')
    data_1000 = data_1000.sort_values('order')
    data_10000 = data_10000.sort_values('order')

    # Add traces for 10 assets
    fig.add_trace(
        go.Bar(
            x=data_10['calculation_type_clean'],
            y=data_10['calculation_time'],
            name='10 Assets',
            marker_color=COLOR_PALETTE['asset_10'],
            text=[f"{val:.3f}s" for val in data_10['calculation_time']],
            textposition='outside',
            offsetgroup=0
        )
    )

    # Add traces for 100 assets
    fig.add_trace(
        go.Bar(
            x=data_100['calculation_type_clean'],
            y=data_100['calculation_time'],
            name='100 Assets',
            marker_color=COLOR_PALETTE['asset_100'],
            text=[f"{val:.3f}s" for val in data_100['calculation_time']],
            textposition='outside',
            offsetgroup=1
        )
    )

    # Add traces for 1000 assets
    fig.add_trace(
        go.Bar(
            x=data_1000['calculation_type_clean'],
            y=data_1000['calculation_time'],
            name='1000 Assets',
            marker_color=COLOR_PALETTE['asset_1000'],
            text=[f"{val:.3f}s" for val in data_1000['calculation_time']],
            textposition='outside',
            offsetgroup=2
        )
    )
    # Add traces for 1000 assets
    fig.add_trace(
        go.Bar(
            x=data_10000['calculation_type_clean'],
            y=data_10000['calculation_time'],
            name='10000 Assets',
            marker_color=COLOR_PALETTE['asset_10000'],
            text=[f"{val:.3f}s" for val in data_10000['calculation_time']],
            textposition='outside',
            offsetgroup=3
        )
    )
    # Update layout
    fig.update_layout(
        title_font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,  # Position legend at top
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        height=410,
        width=900,
        margin=dict(t=10, b=10, l=80, r=40),
        yaxis_title="Calculation Time (seconds)",
        yaxis_tickformat=".3f",
        font=dict(size=14),
        plot_bgcolor='rgba(255, 255, 255, 1)',  # Ensure white background
        paper_bgcolor='rgba(255, 255, 255, 1)'
    )


    # Format y-axis labels
    fig.update_yaxes(
        ticksuffix="s",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)'
    )

    # Create directory for output if it doesn't exist
    output_dir = "paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    pio.write_image(fig, os.path.join(output_dir, 'calculation_times.pdf'))
    pio.write_image(fig, os.path.join(output_dir, 'calculation_times.png'), scale=3)
    plot(fig, filename=os.path.join(output_dir, 'calculation_times.html'), auto_open=False)
    print(f"Calculation time plot saved to {output_dir} folder.")

    return fig


def plot_percentage_flexibility_comparison(aggregation_results):
    """Create a plot comparing flexibility improvement percentages between optimized and sum approaches."""
    print("Creating flexibility percentage comparison plot...")

    # Group data by scenario type and flex pattern
    flexibility_summary = aggregation_results.groupby(
        ['scenario_type', 'flex_pattern', 'num_assets']
    ).agg({
        'opt_flex_range': 'mean',
        'sum_flex_range': 'mean'
    }).reset_index()

    # Calculate percentage improvement
    flexibility_summary['percentage_improvement'] = ((flexibility_summary['opt_flex_range'] -
                                                      flexibility_summary['sum_flex_range']) /
                                                     flexibility_summary['sum_flex_range'] * 100)

    # Handle infinity cases (when sum_flex_range is 0)
    flexibility_summary['percentage_improvement'] = flexibility_summary['percentage_improvement'].replace(
        [float('inf'), -float('inf')], np.nan)

    # Focus on scenarios with 100 assets for cleaner visualization
    plot_data = flexibility_summary#[flexibility_summary['num_assets'] == 100]

    # Simplify pattern names for better readability
    def simplify_pattern(pattern):
        if pattern == 'consumption equal to feed-in':
            return 'Equal Consumption'
        elif pattern == 'higher consumption in one interval':
            return 'One Interval Higher'
        elif pattern == 'higher consumption in random interval':
            return 'Random Interval Higher'
        elif pattern == 'higher consumption or feed-in':
            return 'Higher Cons/Feed-in'
        elif pattern == 'random fixed power':
            return 'Random Power'
        return pattern

    plot_data['simple_pattern'] = plot_data['flex_pattern'].apply(simplify_pattern)

    # Create a subplot with two horizontal panels
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Constant Pricing Scenarios", "Incentive Pricing Scenarios"),
        shared_yaxes=True,
        horizontal_spacing=0.02
    )

    # Plot data for constant scenarios
    constant_data = plot_data[plot_data['scenario_type'] == 'constant']
    constant_data = constant_data.sort_values('percentage_improvement')

    # Plot data for incentive scenarios
    incentive_data = plot_data[plot_data['scenario_type'] == 'incentive']
    incentive_data = incentive_data.sort_values('percentage_improvement')

    # Add traces for constant scenarios
    fig.add_trace(
        go.Bar(
            x=constant_data['simple_pattern'],
            y=constant_data['percentage_improvement'],
            name='Percentage Improvement',
            marker_color=COLOR_PALETTE['optimized'],
            text=[f"{x:.1f}%" for x in constant_data['percentage_improvement']],
            textposition="auto"
        ),
        row=1, col=1
    )

    # Add traces for incentive scenarios
    fig.add_trace(
        go.Bar(
            x=incentive_data['simple_pattern'],
            y=incentive_data['percentage_improvement'],
            name='Percentage Improvement',
            marker_color=COLOR_PALETTE['optimized'],
            text=[f"{x:.1f}%" for x in incentive_data['percentage_improvement']],
            textposition="auto",
            showlegend=False
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        #title_text="Percentage Improvement in Flexibility Range: Optimized vs. Market-Restricted Approach",
        title_font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,  # Position above the plot
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        height=600,
        width=1200,
        margin=dict(t=150, b=100, l=80, r=40),  # Increased top margin for legend
        yaxis_title="Percentage Improvement (%)",
        font=dict(size=14),
        plot_bgcolor='rgba(255, 255, 255, 1)',  # Ensure white background
        paper_bgcolor='rgba(255, 255, 255, 1)'
    )

    # Format y-axis labels
    fig.update_yaxes(
        ticksuffix="%",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)'
    )

    # Create directory for output if it doesn't exist
    output_dir = "paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    pio.write_image(fig, os.path.join(output_dir, 'flexibility_percentage_comparison.pdf'))
    pio.write_image(fig, os.path.join(output_dir, 'flexibility_percentage_comparison.png'), scale=3)
    plot(fig, filename=os.path.join(output_dir, 'flexibility_percentage_comparison.html'), auto_open=False)
    print(f"Flexibility percentage comparison plot saved to {output_dir} folder.")

    return fig

def plot_min_max_flexibility_comparison(aggregation_results):
    """Create a plot comparing min and max flexibility between optimized and sum approaches."""
    print("Creating min/max flexibility comparison plot...")

    # Use the existing num_assets, scenario_type from the main code
    # Filter for 100 assets for cleaner visualization
    aggregation_results_100 = aggregation_results#[aggregation_results['num_assets'] == 100]

    # Create a subplot with two vertical panels
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Minimum Flexibility Comparison", "Maximum Flexibility Comparison"),
        vertical_spacing=0.15
    )

    # For the MIN flexibility comparison
    min_summary = aggregation_results_100.groupby(['scenario_type']).agg({
        'opt_min_flexibility': 'mean',
        'sum_min_flexibility': 'mean'
    }).reset_index()

    # Better labels for pricing scenarios
    min_summary['pricing_label'] = min_summary['scenario_type'].apply(
        lambda x: 'Constant Pricing' if x == 'constant' else 'Incentive Pricing'
    )

    # Sort data for consistent ordering
    min_summary = min_summary.sort_values('pricing_label')

    # For the MAX flexibility comparison
    max_summary = aggregation_results_100.groupby(['scenario_type']).agg({
        'opt_max_flexibility': 'mean',
        'sum_max_flexibility': 'mean'
    }).reset_index()

    # Better labels for pricing scenarios
    max_summary['pricing_label'] = max_summary['scenario_type'].apply(
        lambda x: 'Constant Pricing' if x == 'constant' else 'Incentive Pricing'
    )

    # Sort data for consistent ordering
    max_summary = max_summary.sort_values('pricing_label')

    # Add traces for MIN flexibility
    fig.add_trace(
        go.Bar(
            x=min_summary['pricing_label'],
            y=min_summary['opt_min_flexibility'],
            name='Optimized',
            marker_color=COLOR_PALETTE['optimized'],
            offsetgroup=0
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=min_summary['pricing_label'],
            y=min_summary['sum_min_flexibility'],
            name='Market-Restricted',
            marker_color=COLOR_PALETTE['market_restricted'],
            offsetgroup=1
        ),
        row=1, col=1
    )

    # Add traces for MAX flexibility
    fig.add_trace(
        go.Bar(
            x=max_summary['pricing_label'],
            y=max_summary['opt_max_flexibility'],
            name='Optimized',
            marker_color=COLOR_PALETTE['optimized'],
            offsetgroup=0,
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=max_summary['pricing_label'],
            y=max_summary['sum_max_flexibility'],
            name='Market-Restricted',
            marker_color=COLOR_PALETTE['market_restricted'],
            offsetgroup=1,
            showlegend=False
        ),
        row=2, col=1
    )

    # Add percentage improvements for MIN flexibility
    for i, row in min_summary.iterrows():
        if row['sum_min_flexibility'] == 0:
            improvement = math.inf
        else:
            improvement = ((row['opt_min_flexibility'] - row['sum_min_flexibility']) / abs(row['sum_min_flexibility'])) * 100
        if improvement != math.inf:  # Only show if not infinity
            fig.add_annotation(
                x=row['pricing_label'],
                y=min(row['opt_min_flexibility'], row['sum_min_flexibility']) * 0.9,
                text=f"{improvement:.1f}%",
                showarrow=False,
                font=dict(size=10, color="#333333", family="Arial"),
                row=1, col=1
            )

    # Add percentage improvements for MAX flexibility
    for i, row in max_summary.iterrows():
        if row['sum_max_flexibility'] == 0:
            improvement = math.inf
        else:
            improvement = ((row['opt_max_flexibility'] - row['sum_max_flexibility']) / abs(row['sum_max_flexibility'])) * 100
        if improvement != math.inf:  # Only show if not infinity
            fig.add_annotation(
                x=row['pricing_label'],
                y=max(row['opt_max_flexibility'], row['sum_max_flexibility']) * 1.05,
                text=f"{improvement:.1f}%",
                showarrow=False,
                font=dict(size=10, color="#333333", family="Arial"),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(
        title_text="Min and Max Flexibility Comparison: Optimized vs. Market-Restricted Approach",
        title_font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        height=800,
        width=1000,
        margin=dict(t=150, b=100, l=80, r=40),
        font=dict(size=14),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)'
    )

    # Format y-axis for each subplot
    fig.update_yaxes(
        title="Minimum Flexibility (W)",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)',
        row=1, col=1
    )

    fig.update_yaxes(
        title="Maximum Flexibility (W)",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)',
        row=2, col=1
    )

    # Create directory for output if it doesn't exist
    output_dir = "paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    pio.write_image(fig, os.path.join(output_dir, 'min_max_flexibility_comparison.pdf'))
    pio.write_image(fig, os.path.join(output_dir, 'min_max_flexibility_comparison.png'), scale=3)
    plot(fig, filename=os.path.join(output_dir, 'min_max_flexibility_comparison.html'), auto_open=False)
    print(f"Min/max flexibility comparison plot saved to {output_dir} folder.")

    return fig


def plot_min_max_flexibility_by_pattern(aggregation_results):
    """Create a more detailed plot showing min and max flexibility by flexibility pattern."""
    print("Creating min/max flexibility by pattern comparison plot...")

    # Filter for 100 assets for cleaner visualization
    aggregation_results_100 = aggregation_results#[aggregation_results['num_assets'] == 100]

    # Simplify pattern names for better readability
    def simplify_pattern(pattern):
        if pattern == 'consumption equal to feed-in':
            return 'Equal cons/feed-in'
        elif pattern == 'higher consumption in one interval':
            return 'One Peak'
        elif pattern == 'higher consumption in random interval':
            return 'Random Peak'
        elif pattern == 'higher consumption or feed-in':
            return 'Higher cons/feed-in'
        elif pattern == 'random fixed power':
            return 'Random'
        return pattern

    # Group by flexibility pattern
    pattern_summary = aggregation_results_100.groupby(['flex_pattern']).agg({
        'opt_min_flexibility': 'mean',
        'sum_min_flexibility': 'mean',
        'opt_max_flexibility': 'mean',
        'sum_max_flexibility': 'mean'
    }).reset_index()

    # Apply simplified pattern labels
    pattern_summary['simple_pattern'] = pattern_summary['flex_pattern'].apply(simplify_pattern)

    # Create a subplot with two vertical panels
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Minimum Flexibility by Pattern", "Maximum Flexibility by Pattern"),
        vertical_spacing=0.15
    )

    # Add traces for MIN flexibility
    fig.add_trace(
        go.Bar(
            x=pattern_summary['simple_pattern'],
            y=pattern_summary['opt_min_flexibility'],
            name='Optimized',
            marker_color=COLOR_PALETTE['optimized'],
            offsetgroup=0
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=pattern_summary['simple_pattern'],
            y=pattern_summary['sum_min_flexibility'],
            name='Market-Restricted',
            marker_color=COLOR_PALETTE['market_restricted'],
            offsetgroup=1
        ),
        row=1, col=1
    )

    # Add traces for MAX flexibility
    fig.add_trace(
        go.Bar(
            x=pattern_summary['simple_pattern'],
            y=pattern_summary['opt_max_flexibility'],
            name='Optimized',
            marker_color=COLOR_PALETTE['optimized'],
            offsetgroup=0,
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=pattern_summary['simple_pattern'],
            y=pattern_summary['sum_max_flexibility'],
            name='Market-Restricted',
            marker_color=COLOR_PALETTE['market_restricted'],
            offsetgroup=1,
            showlegend=False
        ),
        row=2, col=1
    )

    # Add percentage improvements for MIN flexibility
    for i, row in pattern_summary.iterrows():
        if row['sum_min_flexibility'] == 0:
            improvement = math.inf
        else:
            improvement = ((row['opt_min_flexibility'] - row['sum_min_flexibility']) / abs(row['sum_min_flexibility'])) * 100
        if improvement != math.inf and abs(improvement) > 1:  # Only show significant differences
            fig.add_annotation(
                x=row['simple_pattern'],
                y=min(row['opt_min_flexibility'], row['sum_min_flexibility']) * 0.9,
                text=f"{improvement:.1f}%",
                showarrow=False,
                font=dict(size=10, color="#333333", family="Arial"),
                row=1, col=1
            )

    # Add percentage improvements for MAX flexibility
    for i, row in pattern_summary.iterrows():
        if row['sum_max_flexibility'] == 0:
            improvement = math.inf
        else:
            improvement = ((row['opt_max_flexibility'] - row['sum_max_flexibility']) / abs(row['sum_max_flexibility'])) * 100
        if improvement != math.inf and abs(improvement) > 1:  # Only show significant differences
            fig.add_annotation(
                x=row['simple_pattern'],
                y=max(row['opt_max_flexibility'], row['sum_max_flexibility']) * 1.05,
                text=f"{improvement:.1f}%",
                showarrow=False,
                font=dict(size=10, color="#333333", family="Arial"),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(
        title_text="Min and Max Flexibility by Pattern: Optimized vs. Market-Restricted Approach",
        title_font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        height=800,
        width=1000,
        margin=dict(t=150, b=100, l=80, r=40),
        font=dict(size=14),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)'
    )

    # Format x-axis for better readability
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=12)
    )

    # Format y-axis for each subplot
    fig.update_yaxes(
        title="Minimum Flexibility (W)",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)',
        row=1, col=1
    )

    fig.update_yaxes(
        title="Maximum Flexibility (W)",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)',
        row=2, col=1
    )

    # Create directory for output if it doesn't exist
    output_dir = "paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    pio.write_image(fig, os.path.join(output_dir, 'min_max_flex_by_pattern.pdf'))
    pio.write_image(fig, os.path.join(output_dir, 'min_max_flex_by_pattern.png'), scale=3)
    plot(fig, filename=os.path.join(output_dir, 'min_max_flex_by_pattern.html'), auto_open=False)
    print(f"Min/max flexibility by pattern plot saved to {output_dir} folder.")

    return fig


def plot_min_max_flexibility_by_assets(aggregation_results):
    """Create a plot showing min and max flexibility by number of assets."""
    print("Creating min/max flexibility by number of assets plot...")

    # Group by number of assets
    assets_summary = aggregation_results.groupby(['num_assets']).agg({
        'opt_min_flexibility': 'mean',
        'sum_min_flexibility': 'mean',
        'opt_max_flexibility': 'mean',
        'sum_max_flexibility': 'mean'
    }).reset_index()

    # Create a subplot with two vertical panels
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Minimum Flexibility by Number of Assets", "Maximum Flexibility by Number of Assets"),
        vertical_spacing=0.15
    )

    # Add traces for MIN flexibility
    fig.add_trace(
        go.Bar(
            x=assets_summary['num_assets'].astype(str),
            y=assets_summary['opt_min_flexibility'],
            name='Optimized',
            marker_color=COLOR_PALETTE['optimized'],
            offsetgroup=0
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=assets_summary['num_assets'].astype(str),
            y=assets_summary['sum_min_flexibility'],
            name='Market-Restricted',
            marker_color=COLOR_PALETTE['market_restricted'],
            offsetgroup=1
        ),
        row=1, col=1
    )

    # Add traces for MAX flexibility
    fig.add_trace(
        go.Bar(
            x=assets_summary['num_assets'].astype(str),
            y=assets_summary['opt_max_flexibility'],
            name='Optimized',
            marker_color=COLOR_PALETTE['optimized'],
            offsetgroup=0,
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=assets_summary['num_assets'].astype(str),
            y=assets_summary['sum_max_flexibility'],
            name='Market-Restricted',
            marker_color=COLOR_PALETTE['market_restricted'],
            offsetgroup=1,
            showlegend=False
        ),
        row=2, col=1
    )

    # Add percentage improvements for MIN flexibility
    for i, row in assets_summary.iterrows():
        if row['sum_min_flexibility'] == 0:
            improvement = math.inf
        else:
            improvement = ((row['opt_min_flexibility'] - row['sum_min_flexibility']) / abs(row['sum_min_flexibility'])) * 100
        if improvement != math.inf:  # Only show if not infinity
            fig.add_annotation(
                x=str(row['num_assets']),
                y=min(row['opt_min_flexibility'], row['sum_min_flexibility']) * 0.9,
                text=f"{improvement:.1f}%",
                showarrow=False,
                font=dict(size=10, color="#333333", family="Arial"),
                row=1, col=1
            )

    # Add percentage improvements for MAX flexibility
    for i, row in assets_summary.iterrows():
        if row['sum_max_flexibility'] == 0:
            improvement = math.inf
        else:
            improvement = ((row['opt_max_flexibility'] - row['sum_max_flexibility']) / abs(row['sum_max_flexibility'])) * 100
        if improvement != math.inf:  # Only show if not infinity
            fig.add_annotation(
                x=str(row['num_assets']),
                y=max(row['opt_max_flexibility'], row['sum_max_flexibility']) * 1.05,
                text=f"{improvement:.1f}%",
                showarrow=False,
                font=dict(size=10, color="#333333", family="Arial"),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(
        title_text="Min and Max Flexibility by Number of Assets: Optimized vs. Market-Restricted Approach",
        title_font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        height=800,
        width=1000,
        margin=dict(t=150, b=100, l=80, r=40),
        font=dict(size=14),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)'
    )

    # Format x-axis
    fig.update_xaxes(
        title="Number of Assets",
        tickfont=dict(size=12)
    )

    # Format y-axis for each subplot
    fig.update_yaxes(
        title="Minimum Flexibility (W)",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)',
        row=1, col=1
    )

    fig.update_yaxes(
        title="Maximum Flexibility (W)",
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.8)',
        row=2, col=1
    )

    # Create directory for output if it doesn't exist
    output_dir = "paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    pio.write_image(fig, os.path.join(output_dir, 'min_max_flex_by_assets.pdf'))
    pio.write_image(fig, os.path.join(output_dir, 'min_max_flex_by_assets.png'), scale=3)
    plot(fig, filename=os.path.join(output_dir, 'min_max_flex_by_assets.html'), auto_open=False)
    print(f"Min/max flexibility by assets plot saved to {output_dir} folder.")

    return fig

def print_summary(aggregation_results):
    print('sum opt min: ', aggregation_results['opt_min_flexibility'].sum())
    print('sum opt max: ', aggregation_results['opt_max_flexibility'].sum())

    print('sum sum min: ', aggregation_results['sum_min_flexibility'].sum())
    print('sum sum max: ', aggregation_results['sum_max_flexibility'].sum())


# Add these functions to the main function
def main():
    try:
        # Load and process data
        aggregation_results, calculation_times = load_and_process_data()

        # Create existing plots
        plot_flexibility_comparison(aggregation_results)
        plot_calculation_times(calculation_times)
        plot_soc_flexibility(aggregation_results)
        plot_percentage_flexibility_comparison(aggregation_results)

        # Create new min/max flexibility plots
        plot_min_max_flexibility_comparison(aggregation_results)
        plot_min_max_flexibility_by_pattern(aggregation_results)
        plot_min_max_flexibility_by_assets(aggregation_results)

        print_summary(aggregation_results)

        print(generate_calculation_time_table(calculation_times))

        print("\nAll plots have been generated successfully!")
        print("Files saved to the 'paper_figures' folder:")
        print("- flexibility_comparison.pdf/.png/.html")
        print("- calculation_times.pdf/.png/.html")
        print("- soc_min_max_flexibility.pdf/.png/.html")
        print("- min_max_flexibility_comparison.pdf/.png/.html")
        print("- min_max_flex_by_pattern.pdf/.png/.html")
        print("- min_max_flex_by_assets.pdf/.png/.html")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure the CSV files are in the 'aggregation_results' folder:")
        print("- aggregation_results/aggregation_results.csv")
        print("- aggregation_results/calculation_times.csv")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()