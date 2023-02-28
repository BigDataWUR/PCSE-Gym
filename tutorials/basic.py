
import collections.abc

#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping


if __name__ == '__main__':

    from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
    from pcse.util import WOFOST80SiteDataProvider

    from pcse_gym.environment.env import PCSEEnv

    env = PCSEEnv(
        model_config='Wofost80_NWLP_FD.conf',
        agro_config='../pcse_gym/environment/configs/agro/potato_cropcalendar.yaml',
        crop_parameters=YAMLCropDataProvider(force_reload=True),
        site_parameters=WOFOST80SiteDataProvider(WAV=10,  # Initial amount of water in total soil profile [cm]
                                                 NAVAILI=10,  # Amount of N available in the pool at initialization of the system [kg/ha]
                                                 PAVAILI=50,  # Amount of P available in the pool at initialization of the system [kg/ha]
                                                 KAVAILI=100,  # Amount of K available in the pool at initialization of the system [kg/ha]
                                                 ),
        soil_parameters=CABOFileReader('../pcse_gym/environment/configs/soil/ec3.CAB'),
    )

    o = env.reset()

    print(o)

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
    #
    # r_sum = 0
    # while not done:
    #     o, r, done, info = env.step(a)
    #     r_sum += r
    #
    # print(o)
    # print(r_sum)
    #
