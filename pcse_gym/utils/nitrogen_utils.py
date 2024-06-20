from typing import Union
import datetime
import pcse

from pcse.soil.snomin import SNOMIN
from pcse.input.csvweatherdataprovider import CSVWeatherDataProvider
from pcse.input.nasapower import NASAPowerWeatherDataProvider

from pcse_gym.envs.common_env import AgroManagementContainer, get_weather_data_provider
from pcse_gym.envs.rewards import get_deposition_amount
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
