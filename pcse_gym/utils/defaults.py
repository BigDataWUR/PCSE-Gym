def get_lintul_default_crop_features():
    # See get_titles() for description of variables
    return ["DVS", "TGROWTH", "LAI", "NUPTT", "TRAN", "TNSOIL", "TRAIN", "TRANRF", "WSO"]


def get_wofost_default_crop_features(pcse_env):
    # See get_titles() for description of variables
    if pcse_env == 1:
        return ["DVS", "TAGP", "LAI", "NuptakeTotal", "TRA", "NAVAIL", "SM", "RFTRA", "WSO"]
    elif pcse_env == 2:
        return ["DVS", "TAGP", "LAI", "NuptakeTotal", "TRA", "NO3", "NH4", "SM", "RFTRA", "WSO"]


def get_wofost_minimal_crop_features(pcse_env):
    # See get_titles() for description of variables
    if pcse_env == 1:
        return ["DVS", "TAGP", "LAI", "NuptakeTotal", "NAVAIL", "SM"]
    elif pcse_env == 2:
        return ["DVS", "TAGP", "LAI", "NuptakeTotal", "NO3", "NH4", "SM"]


def get_wofost_default_po_features():
    # See get_titles() for description of variables
    return ["TAGP", "LAI", "NAVAIL", "NuptakeTotal", "SM"]


def get_default_crop_features(pcse_env=1, minimal=False):
    if pcse_env == 1 and minimal:
        crop_features = get_wofost_minimal_crop_features(pcse_env)
    elif pcse_env == 1:
        crop_features = get_wofost_default_crop_features(pcse_env)
    elif pcse_env == 2 and minimal:
        crop_features = get_wofost_minimal_crop_features(pcse_env)
    elif pcse_env == 2:
        crop_features = get_wofost_default_crop_features(pcse_env)
    else:
        crop_features = get_lintul_default_crop_features()
    return crop_features


def get_default_weather_features():
    # See get_titles() for description of variables
    return ["IRRAD", "TMIN", "RAIN"]


def get_default_action_features():
    return []


def get_default_location():
    return (52, 5.5)


def get_default_years():
    return [*range(1990, 2022)]


def get_default_train_years():
    all_years = get_default_years()
    train_years = [year for year in all_years if year % 2 == 1]
    return train_years


def get_default_test_years():
    all_years = get_default_years()
    test_years = [year for year in all_years if year % 2 == 0]
    return test_years


def get_default_action_space():
    import gymnasium as gym
    action_space = gym.spaces.Discrete(3)
    return action_space