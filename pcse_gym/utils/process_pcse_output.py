import numpy as np


def get_previous_index(pcse_output, timestep):
    return (np.ceil(len(pcse_output) / timestep).astype('int') - 1) * timestep - 1


def get_start_date(pcse_output, timestep):
    return pcse_output[-1 - timestep]['day']


def get_var_names(pcse_output):
    return pcse_output[0].keys()


def get_name_storage_organ(var_names):
    if 'WSO' in var_names:
        return 'WSO'
    elif 'TWSO' in var_names:
        return 'TWSO'
    raise Exception(" (T)WSO not found")


def get_dict_lintul_wofost():
    dict_convert = (("WSO", "TWSO", 10.0), ("TNSOIL", "NAVAIL", 10.0))
    return dict_convert


def needs_conversion(var, dict_lintul_wofost=get_dict_lintul_wofost()):
    for lintul, wofost, conv_factor in dict_lintul_wofost:
        if var == wofost:
            return True
    return False


def get_conversion_factor(var, dict_lintul_wofost=get_dict_lintul_wofost()):
    for lintul, wofost, conv_factor in dict_lintul_wofost:
        if var == lintul:
            return conv_factor
        if var == wofost:
            return 1.0 / conv_factor
    raise Exception(f'{var} not found')


def compute_growth_var(pcse_output, timestep, var):
    var_start = pcse_output[get_previous_index(pcse_output, timestep)][var]
    var_finish = pcse_output[-1][var]
    if var_start is None: var_start = 0.0
    if var_finish is None: var_finish = 0.0
    growth = var_finish - var_start
    return growth


def compute_growth_storage_organ(pcse_output, timestep):
    """
    Computes growth of storage organ in g/m2
    """
    wso_var = get_name_storage_organ(get_var_names(pcse_output))
    wso_growth = compute_growth_var(pcse_output, timestep, wso_var)
    if needs_conversion(wso_var):
        wso_growth = wso_growth * get_conversion_factor(wso_var)
    return wso_growth