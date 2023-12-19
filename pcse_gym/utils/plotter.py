import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def get_cumulative_variables():
    return ['fertilizer', 'reward']

def get_titles():
    def def_value(): return ("", "")

    return_dict = defaultdict(def_value)
    return_dict["DVS"] = ("Development stage", "-")
    return_dict["TGROWTH"] = ("Total biomass (above and below ground)", "g/m2")
    return_dict["LAI"] = ("Leaf area Index", "-")
    return_dict["NUPTT"] = ("Total nitrogen uptake", "gN/m2")
    return_dict["TRAN"] = ("Transpiration", "mm/day")
    return_dict["TIRRIG"] = ("Total irrigation", "mm")
    return_dict["TNSOIL"] = ("Total soil inorganic nitrogen", "gN/m2")
    return_dict["TRAIN"] = ("Total rainfall", "mm")
    return_dict["TRANRF"] = ("Transpiration reduction factor", "-")
    return_dict["TRUNOF"] = ("Total runoff", "mm")
    return_dict["TAGBM"] = ("Total aboveground biomass", "g/m2")
    return_dict["TTRAN"] = ("Total transpiration", "mm")
    return_dict["WC"] = ("Soil water content", "m3/m3")
    return_dict["WLVD"] = ("Weight dead leaves", "g/m2")
    return_dict["WLVG"] = ("Weight green leaves", "g/m2")
    return_dict["WRT"] = ("Weight roots", "g/m2")
    return_dict["WSO"] = ("Weight storage organs", "g/m2")
    return_dict["TWSO"] = ("Weight storage organs", "kg/ha")
    return_dict["WST"] = ("Weight stems", "g/m2")
    return_dict["TGROWTHr"] = ("Growth rate", "g/m2/day")
    return_dict["NRF"] = ("Nitrogen reduction factor", "-")
    return_dict["GRF"] = ("Growth reduction factor", "-")

    return_dict["DVS"] = ("Development stage", "-")
    return_dict["TAGP"] = ("Total above-ground Production", "kg/ha")
    return_dict["LAI"] = ("Leaf area Index", "-")
    return_dict["RNuptake"] = ("Total nitrogen uptake", "kgN/ha")
    return_dict["TRA"] = ("Transpiration", "cm/day")
    return_dict["NAVAIL"] = ("Total soil inorganic nitrogen", "kgN/ha")
    return_dict["SM"] = ("Volumetric soil moisture content", "-")
    return_dict["RFTRA"] = ("Transpiration reduction factor", "-")
    return_dict["TRUNOF"] = ("Total runoff", "mm")
    return_dict["TAGBM"] = ("Total aboveground biomass", "kg/ha")
    return_dict["TTRAN"] = ("Total transpiration", "mm")
    return_dict["WC"] = ("Soil water content", "m3/m3")
    return_dict["Ndemand"] = ("Total N demand of crop", "kgN/ha")
    return_dict["NuptakeTotal"] = ("Total N uptake of crop", "kgN/ha/d")
    return_dict["FERT_N_SUPPLY"] = ("Total N supplied by actions", "kgN/ha")

    return_dict["fertilizer"] = ("Nitrogen application", "kg/ha")
    return_dict["TMIN"] = ("Minimum temperature", "°C")
    return_dict["TMAX"] = ("Maximum temperature", "°C")
    return_dict["IRRAD"] = ("Incoming global radiation", "J/m2/day")
    return_dict["RAIN"] = ("Daily rainfall", "cm/day")

    return return_dict


def plot_year_loc_heatmap(results_dict, variable, year_locs, cumulative_variables=get_cumulative_variables(), ax=None, fig=None, ylim=None,
                  put_legend=True):
    titles = get_titles()
    xmax = 0
    xmin = 9999
    check = []

    # generator for date of year
    def doy_sender(_i):
        gen.send(None)
        return gen.send(_i)

    # structure x using DVS
    for label, results in results_dict.items():
        x, _ = zip(*results[0]['DVS'].items())
        x = ([i.timetuple().tm_yday for i in x])
        inc = all(i < j for i, j in zip(x, x[1:]))
        if not inc:
            x = restructure_x(x)
        if max(x) > xmax: xmax = max(x)
        if min(x) < xmin: xmin = min(x)

    # make heatmap
    if isinstance(variable, list):
        dataframes_list = pd.DataFrame()
        for v in variable:
            gen = doy_generator()
            df = pd.DataFrame.from_dict(results_dict[year_locs][0][v], orient='index', columns=[v])
            df = df.rename(lambda i: doy_sender(i.timetuple().tm_yday))
            dataframes_list = pd.concat([dataframes_list, df], axis=1)

        dataframes_list = dataframes_list.T

        left, right, down, up = dataframes_list.columns[0], dataframes_list.columns[-1], 0, len(dataframes_list.index)
        extent = [left-6, right+6, down-0.5, up-0.5]

        # plot heatmap
        heatmap = ax.imshow(dataframes_list, aspect='auto', cmap='RdYlBu', extent=extent, origin='lower')

        print(dataframes_list)
        ax.set_yticks(np.arange(0, len(dataframes_list.index)))
        ax.set_yticklabels(dataframes_list.index)
        ax.yaxis.grid(which="minor", color="black", linestyle=':', linewidth=1)

        ax.set_title('measuring actions', fontsize="8.5")


    else:
        gen = doy_generator()  # restart the generator
        df = pd.DataFrame.from_dict(results_dict[year_locs][0][variable], orient='index', columns=[variable])
        df = df.rename(lambda i: doy_sender(i.timetuple().tm_yday))

        ax.step(df.index, df[variable], where='post')

        ax.axhline(y=0, color='lightgrey', zorder=1)
        ax.margins(x=0)



    from matplotlib.ticker import FixedLocator
    ax.xaxis.set_minor_locator(FixedLocator(range(0, xmax, 7)))
    ax.xaxis.grid(True, which='minor')
    ax.tick_params(axis='x', which='minor', grid_alpha=0.7, colors=ax.get_figure().get_facecolor(), grid_ls=":")

    months, month_days = ticks_checker(inc, xmin, xmax)
    ax.set_xticks(month_days)
    ax.set_xticklabels(months)

    if isinstance(variable, list):
        name, unit = "crop variables", ""
        ax.set_title(f"{year_locs} - {name}", fontsize='8.5')
    else:
        name, unit = titles[variable]
        ax.set_title(f"{variable} - {name}", fontsize='8.5')
        ax.set_ylabel(f"[{unit}]")
    if variable in cumulative_variables:
        ax.set_title(f"{variable} (cumulative) - {name}")

    return ax

def restructure_x(day_nums):
    # sanity check
    # if number resets to 1, add subsequent number with previous so on
    offset = 0
    new_num = []
    for i, n in enumerate(day_nums):
        if i > 0 and day_nums[i] < day_nums[i - 1]:
            offset += day_nums[i - 1]
        new_num.append(n + offset)
    return new_num

def month_of_year_ind(day_of_year):
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] * 2
    cumulative_days = 0
    for i, length in enumerate(month_lengths):
        cumulative_days += length
        # sanity check
        # if our day_of_year is less than the cumulative days,
        # we've found our month
        if day_of_year <= cumulative_days:
            return i
    else:
        return None

def ticks_checker(inc_flag, _xmin, xmax):
    mons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    mon_days = [0, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335,
                366, 397, 425, 456, 486, 517, 547, 578, 609, 639, 670, 700]
    if not inc_flag:
        mons = mons * 2
        _extra_month = next(m[0] for m in enumerate(mon_days) if m[1] >= xmax)
        _xmin = month_of_year_ind(_xmin)
        mon_days = mon_days[_xmin:_extra_month + 1]
        mons = mons[_xmin:_extra_month + 1]
    else:
        _extra_month = next(m[0] for m in enumerate(mon_days) if m[1] >= xmax)
        mon_days = mon_days[0:_extra_month + 1]
        mons = mons[0:_extra_month + 1]
    return mons, mon_days

def doy_generator():
    # sanity check
    # starts in Oct (274), resets in Jan (1), continue with offset for rest
    # new dataframe starts in Oct (274)
    last_value = None
    while True:
        current_value = (yield)
        if last_value is None or (last_value < current_value):
            last_value = current_value
        elif last_value > current_value:
            offset = last_value
            current_value += offset
        yield current_value

def plot_variable(results_dict, variable='reward', cumulative_variables=get_cumulative_variables(), ax=None, ylim=None,
                  put_legend=True, plot_average=False, plot_heatmap=False):
    titles = get_titles()
    xmax = 0
    xmin = 9999
    check = []

    # generator for date of year
    def doy_sender(_i):
        gen.send(None)
        return gen.send(_i)

    # structure x and y ticks
    for label, results in results_dict.items():
        x, y = zip(*results[0][variable].items())
        x = ([i.timetuple().tm_yday for i in x])
        inc = all(i < j for i, j in zip(x, x[1:]))
        if not inc:
            x = restructure_x(x)
        if variable in cumulative_variables: y = np.cumsum(y)
        if max(x) > xmax: xmax = max(x)
        if min(x) < xmin: xmin = min(x)
        if not plot_average:
            ax.step(x, y, label=label, where='post')


    if plot_average:
        dataframes_list = []
        for label, results in results_dict.items():
            gen = doy_generator()  # restart the generator
            df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
            df = df.rename(lambda i: doy_sender(i.timetuple().tm_yday))
            dataframes_list.append(df)

        plot_df = pd.concat(dataframes_list, axis=1)
        plot_df.sort_index(inplace=True)
        if variable in cumulative_variables: plot_df = plot_df.apply(np.cumsum, axis=0)
        if variable.startswith("measure"):
            plot_df.ffill(inplace=True)
            ax.step(plot_df.index, plot_df.sum(axis=1), 'k-', where='post')
            ax.fill_between(plot_df.index, plot_df.min(axis=1), plot_df.sum(axis=1), color='g', step='post')
        elif variable == 'action':
            plot_df.fillna(0, inplace=True)
            ax.step(plot_df.index, plot_df.median(axis=1), 'k-', where='post')
            ax.fill_between(plot_df.index, plot_df.quantile(0.25, axis=1), plot_df.quantile(0.75, axis=1), step='post')
        else:
            plot_df.ffill(axis=0, inplace=True)
            ax.step(plot_df.index, plot_df.median(axis=1), 'k-', where='post')
            ax.fill_between(plot_df.index, plot_df.quantile(0.25, axis=1), plot_df.quantile(0.75, axis=1), step='post')

    ax.axhline(y=0, color='lightgrey', zorder=1)
    ax.margins(x=0)

    from matplotlib.ticker import FixedLocator
    ax.xaxis.set_minor_locator(FixedLocator(range(0, xmax, 7)))
    ax.xaxis.grid(True, which='minor')
    ax.tick_params(axis='x', which='minor', grid_alpha=0.7, colors=ax.get_figure().get_facecolor(), grid_ls=":")

    months, month_days = ticks_checker(inc, xmin, xmax)
    ax.set_xticks(month_days)
    ax.set_xticklabels(months)

    name, unit = titles[variable]
    ax.set_title(f"{variable} - {name}")
    if variable in cumulative_variables:
        ax.set_title(f"{variable} (cumulative) - {name}")
    ax.set_ylabel(f"[{unit}]")
    if ylim is not None:
        ax.set_ylim(ylim)
    if put_legend:
        ax.legend()
    else:
        ax.legend()
        ax.get_legend().set_visible(False)
    return ax


def plot_var_vs_freq(results_dict, variable='measure_LAI', ax=None, ylim=None,
                     put_legend=True, n_year_loc=32):
    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        dataframes_list.append(df)

    plot_measure = pd.concat(dataframes_list, axis=1)
    plot_measure = plot_measure.sum(axis=1)

    variable = variable.replace("measure_", "")

    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        dataframes_list.append(df)

    plot_var = pd.concat(dataframes_list, axis=1)
    plot_var = plot_var.std(ddof=0, axis=1)

    plot_df = pd.concat([plot_measure, plot_var], axis=1)
    plot_df.dropna(inplace=True)
    plot_df = plot_df.rename(columns={0: 'measure', 1: 'variance'})

    ax.scatter(plot_df['measure'], plot_df['variance'], c='green', alpha=0.5, linewidths=1, edgecolors='black')

    ax.set_title(f'measuring actions vs variance for {variable}')

    ax.set_xticks(np.arange(0, n_year_loc + 1))
    ax.set_xlabel(f'measuring frequency across years and locations')

    ax.set_ylabel(f'variance across years and locations')

    return ax


def plot_var_vs_freq_box(results_dict, variable='measure_LAI', ax=None, ylim=None,
                  put_legend=True, n_year_loc=32):

    titles = get_titles()

    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        df.index = df.index.map(lambda d: d.strftime('%m-%d'))
        dataframes_list.append(df)

    plot_measure = pd.concat(dataframes_list, axis=1)
    plot_measure = plot_measure.sum(axis=1)

    variable = variable.replace("measure_","")

    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        df.index = df.index.map(lambda d: d.strftime('%m-%d'))
        dataframes_list.append(df)

    plot_var = pd.concat(dataframes_list, axis=1)

    plot_df = pd.concat([plot_measure, plot_var], axis=1)
   # plot_df.dropna(inplace=True)
    plot_df = plot_df.rename(columns={0:'measure'})

    col1_values = range(int(max(plot_df['measure']))+1)

    # Initialize a dictionary to hold boxplot data
    boxplot_data = {i: [] for i in col1_values}

    for col in plot_df.columns[1:]:
        for val in col1_values:
            matched_data = plot_df[plot_df['measure'] == val][col].dropna()
            if not matched_data.empty:
                boxplot_data[val].extend(matched_data)

    bp = ax.boxplot(boxplot_data.values(), patch_artist=True, positions=list(boxplot_data.keys()))

    colors = plt.cm.viridis(np.linspace(0, 1, len(boxplot_data)))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for flier in bp['fliers']:
        flier.set(marker='o', color='black', alpha=0.5)

    ax.grid(True, linestyle='--', alpha=0.7)

    name, unit = titles[variable]
    ax.set_title(f'measuring actions for {variable}')

    ax.set_xticklabels(list(boxplot_data.keys()), rotation=315, fontsize=7)
    ax.set_xlabel(f'measuring frequency')

    ax.set_ylabel(f'{variable} [{unit}] variance across years and locations')

    return ax

def plot_var_vs_freq_scatter(results_dict, variable='measure_LAI', ax=None):

    # Function to find the nearest weekly date
    def nearest_weekly_date(date, base_date):
        days_difference = (date - base_date).days
        nearest_weekly_difference = round(days_difference / 7) * 7
        return base_date + pd.Timedelta(days=nearest_weekly_difference)

    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        df.index = df.index.map(lambda d: d.strftime('%m-%d'))
        dataframes_list.append(df)

    plot_measure = pd.concat(dataframes_list, axis=1)
    plot_measure = plot_measure.sum(axis=1)

    variable = variable.replace("measure_","")

    dataframes_list = []
    for label, results in results_dict.items():
        df = pd.DataFrame.from_dict(results[0][variable], orient='index', columns=[label])
        df.index = df.index.map(lambda d: d.strftime('%m-%d'))
        dataframes_list.append(df)

    plot_var = pd.concat(dataframes_list, axis=1)
    plot_var = plot_var.std(ddof=0, axis=1)

    plot_df = pd.concat([plot_var, plot_measure], axis=1)
    plot_df.dropna(inplace=True)
    plot_df = plot_df.rename(columns={0:'variance', 1: 'measure'})
    # plot_df = plot_df.sort_values(by=['variance'])

    plot_df['Date'] = plot_df.index

    # Convert date strings to datetime objects (considering a non-leap year)
    base_year = 2021  # placeholder non-leap year
    base_date = pd.to_datetime(f'{base_year}-10-01')  # Starting date (1st of October)
    plot_df['Date'] = pd.to_datetime(plot_df['Date'] + f'-{base_year}', format='%m-%d-%Y')

    # Apply the function to assign each date to the nearest weekly date
    plot_df['Nearest_Weekly_Date'] = plot_df['Date'].apply(lambda d: nearest_weekly_date(d, base_date))
    result = plot_df.groupby('Nearest_Weekly_Date').agg({'variance': 'mean', 'measure': 'sum'})

    # Convert the index back to MM-DD format
    result.index = result.index.map(lambda d: d.strftime('%m-%d'))
    result = result.sort_values(by=['variance'])

    result['variance'] = (result['variance'] - result['variance'].min()) / (result['variance'].max() - result['variance'].min())

    ax.scatter(result['variance'], result['measure'], edgecolor='black', alpha=0.7)

    titles = get_titles()

    ax.set_xticks(np.arange(min(result['variance']), max(result['variance'])+0.1, 0.1))

    # ax.set_xticklabels(range(0, int(max(result['variance']))), rotation=315, fontsize=8)

    name, unit = titles[variable]

    ax.set_ylim([-1, 33])

    ax.set_title(f'measuring actions for {variable}', fontsize=10, fontweight='bold', color='green')

    ax.set_ylabel(f'measuring frequency between years', fontsize=8)

    ax.set_xlabel(f'Normalized {variable} [{unit}] variance across years and locations', fontsize=8)

    return ax

