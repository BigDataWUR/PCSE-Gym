import calendar
from typing import Union
import datetime
import pcse

from pcse.soil.snomin import SNOMIN
from pcse.input.csvweatherdataprovider import CSVWeatherDataProvider
from pcse.input.nasapower import NASAPowerWeatherDataProvider

from pcse_gym.envs.common_env import AgroManagementContainer, get_weather_data_provider
from pcse_gym.utils.weather_utils.weather_functions import generate_date_list

mg_to_kg = 1e-6
L_to_m3 = 1e-3
m2_to_ha = 1e-4


def map_random_to_real_year(y_rand, test_year_start=1990, test_year_end=2022, train_year_start=4000,
                            train_year_end=5999):
    # This is a simple linear mapping to convert fake year into real year
    assert y_rand in range(train_year_start, train_year_end + 1)
    y_real = (test_year_start + (y_rand - train_year_start) * (test_year_end - test_year_start)
              / (train_year_end - train_year_start))
    return int(y_real)


def calculate_year_n_deposition(
        year: int,
        loc: tuple,
        agmt: AgroManagementContainer,
        site_params: dict,
        random_weather: bool = False,
) -> tuple[float, float]:
    assert year == agmt.crop_end_date.year

    nh4concentration_r = site_params['NH4ConcR']
    no3concentration_r = site_params['NO3ConcR']

    growing_dates = [agmt.crop_start_date + datetime.timedelta(days=x)
                     for x
                     in range((agmt.crop_end_date - agmt.crop_start_date).days + 1)
                     ]
    no3depo_year = 0.0
    nh4depo_year = 0.0
    conv = 1

    wdp = get_weather_data_provider(loc, random_weather)
    if isinstance(wdp, NASAPowerWeatherDataProvider):
        conv = 10
    elif isinstance(wdp, CSVWeatherDataProvider):
        conv = 1
    for date in growing_dates:
        # sanity check
        # rain in mm, equivalent to L/m2
        # no3conc in mg/L
        # no3depo has to be in kg/ha

        nh4depo_day = wdp(date).RAIN * conv * nh4concentration_r * mg_to_kg / m2_to_ha
        no3depo_day = wdp(date).RAIN * conv * no3concentration_r * mg_to_kg / m2_to_ha
        nh4depo_year += nh4depo_day
        no3depo_year += no3depo_day

    return nh4depo_year, no3depo_year


def calculate_day_n_deposition(
        day_rain: float,  # in mm
        site_params: dict,
):
    """
    Function to calculate daily NO3 and NH4 deposition amount, given rain that day

    :param site_params:
    :param day_rain:
    :return:
    """
    nh4concentration_r = site_params['NH4ConcR']
    no3concentration_r = site_params['NO3ConcR']

    nh4_day_depo = day_rain * nh4concentration_r * mg_to_kg / m2_to_ha
    no3_day_depo = day_rain * no3concentration_r * mg_to_kg / m2_to_ha

    return nh4_day_depo, no3_day_depo


def aggregate_n_depo_days(
    timestep: int,
    day_rain: list[float],
    site_params: dict,
) -> tuple[float, float]:
    aggregated_nh4_depo, aggregated_no3_depo = 0.0, 0.0
    for _, rain in zip(range(1, timestep + 1), day_rain):
        nh4_day_depo, no3_day_depo = calculate_day_n_deposition(rain, site_params)
        aggregated_nh4_depo += nh4_day_depo
        aggregated_no3_depo += no3_day_depo

    return aggregated_nh4_depo, aggregated_no3_depo



def convert_year_to_n_concentration(year: int,
                                    loc: tuple = (52.0, 5.5),
                                    random_weather: bool = False) -> tuple[float, float]:
    wdp = get_weather_data_provider(loc, random_weather)

    nh4_year, no3_year = get_deposition_amount(map_random_to_real_year(year) if random_weather else year)

    daily_year_dates = generate_date_list(datetime.date(year, 1, 1), datetime.date(year, 12, 31))

    rain_year = None

    if isinstance(wdp, NASAPowerWeatherDataProvider):
        rain_year = sum([wdp(day).RAIN * 10 for day in daily_year_dates])
    elif isinstance(wdp, CSVWeatherDataProvider):
        rain_year = sum([wdp(day).RAIN for day in daily_year_dates])

    print(rain_year)

    # sanity check
    # deposition amount is kg / ha
    # rain is in mm ~ L/m2
    # nxConcR need to be in mg / L

    nh4_conc_r = nh4_year * ((1 / mg_to_kg) / (1 / m2_to_ha)) / rain_year
    no3_conc_r = no3_year * ((1 / mg_to_kg) / (1 / m2_to_ha)) / rain_year

    return nh4_conc_r, no3_conc_r


def get_deposition_amount(year) -> tuple:
    """Currently only supports amount from the Netherlands"""
    if year is None or 1900 < year > 2030:
        NO3 = 3
        NH4 = 9
    else:
        ''' Linear functions of N deposition based on
            data in the Netherlands from CLO (2022)'''
        NO3 = 538.868 - 0.264 * year
        NH4 = 697 - 0.339 * year

    return NH4, NO3


def get_disaggregated_deposition(year, start_date, end_date):
    """
    Function to linearly disaggregate annual N deposition amount
    """

    assert start_date < end_date

    if start_date.year != end_date.year:
        nh4_s, no3_s = get_disaggregated_deposition(start_date.year, start_date,
                                                    datetime.date(year=start_date.year, month=12, day=31))
        nh4_e, no3_e = get_disaggregated_deposition(end_date.year, datetime.date(year=end_date.year, month=1, day=1),
                                                    end_date)
        nh4_dis = nh4_s + nh4_e
        no3_dis = no3_s + no3_e

        return nh4_dis, no3_dis

    date_range = (end_date - start_date).days

    nh4_full, no3_full = get_deposition_amount(year)

    daily_nh4 = nh4_full / get_days_in_year(year)
    daily_no3 = no3_full / get_days_in_year(year)

    nh4_dis = daily_nh4 * date_range
    no3_dis = daily_no3 * date_range

    return nh4_dis, no3_dis


def get_days_in_year(year):
    return 365 + calendar.isleap(year)


def input_nue(n_input, year=None, start=None, end=None, n_seed=3.5):
    if start is None or end is None:
        nh4, no3 = get_deposition_amount(year)
    else:
        if year < 2500:
            nh4, no3 = get_disaggregated_deposition(year=year, start_date=start, end_date=end)
        else:
            nh4, no3 = get_deposition_amount(year)
    n_depo = nh4 + no3
    return n_input + n_seed + n_depo


def get_surplus_n(n_input, n_so, year=None, start=None, end=None, n_seed=3.5):
    n_i = input_nue(n_input, year=year, start=start, end=end, n_seed=n_seed)

    return n_i - n_so
