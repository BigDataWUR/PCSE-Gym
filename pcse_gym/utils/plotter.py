import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def get_cumulative_variables():
    return ['fertilizer', 'reward']

def get_ylim_dict(n=32, wso=0):
    def def_value():
        return None

    if n == 0:
        n = 32

    ylim = defaultdict(def_value)
    ylim['WSO'] = [0, 10000 if wso==1 else 1000]
    ylim['TWSO'] = [0, 10000]
    ylim['measure_SM'] = [0, n]
    ylim['measure_TAGP'] = [0, n]
    ylim['measure_random'] = [0, n]
    ylim['measure_LAI'] = [0, n]
    ylim['measure_NuptakeTotal'] = [0, n]
    ylim['measure_NAVAIL'] = [0, n]
    ylim['measure_SM'] = [0, n]
    ylim['measure'] = [0, n]
    ylim['prob_SM'] = [0, 1.0]
    ylim['prob_TAGP'] = [0, 1.0]
    ylim['prob_random'] = [0, 1.0]
    ylim['prob_LAI'] = [0, 1.0]
    ylim['prob_NuptakeTotal'] = [0, 1.0]
    ylim['prob_NAVAIL'] = [0, 1.0]
    ylim['prob_SM'] = [0, 1.0]
    ylim['prob_measure'] = [0, 1.0]
    return ylim

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

    where = 'post'

    if plot_average:
        # get top soil layer
        if variable in ['SM', 'NH4', 'NO3', 'WC']:
            for k, v in results_dict.items():
                for key in v[0][variable].keys():
                    results_dict[k][0][variable][key] = results_dict[k][0][variable][key][0]
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
            ax.step(plot_df.index, plot_df.sum(axis=1), 'k-', where=where)
            ax.fill_between(plot_df.index, plot_df.min(axis=1), plot_df.sum(axis=1), color='g', step=where)
        elif variable == 'action':
            plot_df.fillna(0, inplace=True)
            ax.step(plot_df.index, plot_df.median(axis=1), 'k-', where=where)
            ax.fill_between(plot_df.index, plot_df.quantile(0.25, axis=1), plot_df.quantile(0.75, axis=1), step=where)
        else:
            plot_df.ffill(axis=0, inplace=True)
            ax.step(plot_df.index, plot_df.median(axis=1), 'k-', where=where)
            ax.fill_between(plot_df.index, plot_df.quantile(0.25, axis=1), plot_df.quantile(0.75, axis=1), step=where)

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
