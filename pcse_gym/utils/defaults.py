import torch
import gymnasium as gym
import numpy as np


def get_lintul_default_crop_features():
    # See helper.get_titles() for description of variables
    return ["DVS", "TGROWTH", "LAI", "NUPTT", "TRAN", "TNSOIL", "TRAIN", "TRANRF", "WSO"]


def get_wofost_default_crop_features():
    return ["DVS", "TAGP", "LAI", "NuptakeTotal", "TRA", "NAVAIL", "SM", "RFTRA", "TWSO"]


def get_default_weather_features():
    # See helper.get_titles() for description of variables
    return ["IRRAD", "TMIN", "RAIN"]


def get_default_action_features():
    return []


def get_default_location():
    return (52, 5.5)


def get_default_train_years():
    all_years = [*range(1990, 2022)]
    train_years = [year for year in all_years if year % 2 == 1]
    return train_years


def get_default_test_years():
    all_years = [*range(1990, 2022)]
    test_years = [year for year in all_years if year % 2 == 0]
    return test_years


def get_test_tensor(crop_features=get_wofost_default_crop_features(), action_features=get_default_action_features(),
                    weather_features=get_default_weather_features(), n_days=7):
    test_tensor = torch.zeros(2, n_days * len(weather_features) + len(crop_features) + len(action_features))
    for d in range(n_days):
        for i in range(len(weather_features)):
            j = d * len(weather_features) + len(crop_features) + len(action_features) + i
            test_tensor[0, j] = test_tensor[0, j] + d + i * 0.1
            test_tensor[1, j] = test_tensor[1, j] + 100 + d + i * 0.1
    return test_tensor


def get_multi_discrete_action_space(action_space, kwarg):
    assert isinstance(action_space, gym.spaces.Discrete)
    md = list(np.full(len(kwarg), 2))

    len_action_space = [action_space.n]
    return gym.spaces.MultiDiscrete(len_action_space+md)



