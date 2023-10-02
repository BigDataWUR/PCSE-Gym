from pcse.db.nasapower import NASAPowerWeatherDataProvider
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os

NL_loc_ext = [(52, 5.5), (51.5, 5), (52.5, 6.0)]
LT_loc_ext = [(55.0, 23.5), (55.0, 24.0), (55.5, 23.5)]

NL_loc = (52, 5.5)
LT_loc = (55.0, 23.5)

wdp = NASAPowerWeatherDataProvider(*NL_loc)


def plot_weather_data(weather_data, variables: list, start_year=None, end_year=None):
    """
    Plots the specified variables over the given time range.

    Parameters:
    - variables: List of variables to plot (e.g., ['IRRAD', 'TMIN', 'RAIN'])
    - start_year: Start year for the range. If None, will use the first year in the data.
    - end_year: End year for the range. If None, will use the last year in the data.
    """

    filtered_data = filter_years(weather_data, start_year, end_year)

    dates = [entry['DAY'] for entry in filtered_data]

    # Plotting
    plt.figure(figsize=(14, 7))
    for var in variables:
        values = [entry[var] for entry in filtered_data]
        plt.plot(dates, values, label=var)

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(
        f"Daily Change of {', '.join(variables)} from {start_year if start_year else min(dates).year} to {end_year if end_year else max(dates).year}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_weather_data_weekly(weather_data, variables: list, start_year=None, end_year=None, aggregation='mean', save=False):
    """
    Plots the specified variables over the given time range with a weekly timestep.

    Parameters:
    - variables: List of variables to plot (e.g., ['IRRAD', 'TMIN', 'RAIN'])
    - start_year: Start year for the range. If None, will use the first year in the data.
    - end_year: End year for the range. If None, will use the last year in the data.
    - aggregation: Method of aggregation ('mean' or 'sum')
    """

    filtered_data = filter_years(weather_data, start_year, end_year)

    # Group by year and week number
    weekly_data = defaultdict(list)
    for entry in filtered_data:
        year, week, _ = entry['DAY'].isocalendar()
        weekly_data[(year, week)].append(entry)

    aggregated_data = {}
    for (year, week), entries in weekly_data.items():
        aggregated_entry = {'DAY': f"{year}-W{week}"}
        for var in variables:
            if aggregation == 'mean':
                aggregated_entry[var] = np.mean([entry[var] for entry in entries])
            elif aggregation == 'sum':
                aggregated_entry[var] = np.sum([entry[var] for entry in entries])
        aggregated_data[(year, week)] = aggregated_entry

    # Extract week labels and values for the specified variables
    weeks = sorted(aggregated_data.keys())
    week_labels = [f"{year}-W{week}" for year, week in weeks]

    # Plotting
    plt.figure(figsize=(21, 14))
    for var in variables:
        values = [aggregated_data[week][var] for week in weeks]
        plt.plot(week_labels, values, label=var)

    plt.xlabel("Week")
    plt.ylabel("Value")
    plt.title(
        f"Weekly {aggregation.capitalize()} of {', '.join(variables)} from {start_year if start_year else min(weeks)[0]} to {end_year if end_year else max(weeks)[0]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ticks = 1 if end_year - start_year < 3 else 52
    plt.xticks(week_labels[::ticks], rotation=45)
    if save:
        plt.savefig(os.path.abspath(f"NASAPower_{variables[0]}_{start_year}_to_{end_year}"))
    else:
        plt.show()


def plot_yearly_extremes(weather_data, variable: str, start_year=None, end_year=None, save=False):
    """
    Plots the maximum and minimum values of the specified variable for each year.

    Parameters:
    - variable: Variable to plot (e.g., 'IRRAD')
    - start_year: Start year for the range. If None, will use the first year in the data.
    - end_year: End year for the range. If None, will use the last year in the data.
    """

    # Filter the data based on the specified time range
    filtered_data = filter_years(weather_data, start_year, end_year)

    # Group by year
    yearly_data = defaultdict(list)
    for entry in filtered_data:
        year = entry['DAY'].year
        yearly_data[year].append(entry)

    # Find yearly extremes
    years = sorted(yearly_data.keys())
    max_values = [max([entry[variable] for entry in yearly_data[year]]) for year in years]
    min_values = [min([entry[variable] for entry in yearly_data[year]]) for year in years]
    year_labels = [f"{year}" for year in years]

    plt.figure(figsize=(14, 7))
    plt.plot(years, max_values, label=f'Max {variable}', marker='o', linestyle='-')
    plt.plot(years, min_values, label=f'Min {variable}', marker='o', linestyle='-')

    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title(f"Yearly Extremes of {variable}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.xticks(year_labels[::], rotation=45)
    if save:
        plt.savefig(os.path.abspath(f"NASAPower_extremes_{variable}_{start_year}_to_{end_year}"))
    else:
        plt.show()


def filter_years(weather_data, start_year, end_year):
    filtered = [
        entry for entry in weather_data
        if (start_year is None or entry['DAY'].year >= start_year) and
           (end_year is None or entry['DAY'].year <= end_year)
    ]
    return filtered


def get_some_plots():
    w_data = wdp.export()
    plot_weather_data_weekly(w_data, ['IRRAD'], 1990, 2022, save=True)
    plot_weather_data_weekly(w_data, ['RAIN'], 1990, 2022, save=True)
    plot_weather_data_weekly(w_data, ['TMIN'], 1990, 2022, save=True)

    plot_yearly_extremes(w_data, 'IRRAD', 1990, 2022, save=True)
    plot_yearly_extremes(w_data, 'RAIN', 1990, 2022, save=True)
    plot_yearly_extremes(w_data, 'TMIN', 1990, 2022, save=True)

