
import collections.abc

#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping


if __name__ == '__main__':

    from pcse.fileinput import CABOFileReader
    from pcse.util import WOFOST72SiteDataProvider

    from pcse_gym.environment.env import PCSEEnv

    env = PCSEEnv(
        model_config='Wofost72_WLP_FD.conf',
        agro_config='../pcse_gym/environment/configs/agro/sugarbeet_calendar.yaml',
        crop_parameters=CABOFileReader('../pcse_gym/environment/configs/crop/SUG0601.CAB'),
        site_parameters=WOFOST72SiteDataProvider(WAV=10),
        soil_parameters=CABOFileReader('../pcse_gym/environment/configs/soil/ec3.CAB'),
    )

    o = env.reset()

    # print(o)

    # Define an action that does nothing
    a = {
        'irrigation': 0,
        'N': 0,
        'P': 0,
        'K': 0,
    }

    o, r, done, info = env.step(a)

    # print(o)
    # print(r)

    r_sum = 0
    while not done:
        o, r, done, info = env.step(a)
        r_sum += r

    print(o)
    print(r_sum)

