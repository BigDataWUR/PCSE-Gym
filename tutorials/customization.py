
import collections.abc

#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping


if __name__ == '__main__':

    from pcse_gym.environment.env import PCSEEnv


    class CustomPCSEEnv(PCSEEnv):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # N price per unit
            self._n_price = 2
            # Yield price per unit
            self._y_price = 1

            # N application costs
            self._na_price = 10
            # Keep track of how much nitrogen has been applied in the last time step
            self._na = 0

        def _apply_action(self, action):
            super()._apply_action(action)
            # Keep track of the amount of nitrogen that was applied
            self._na = action.get('N', 0)

        def _get_reward(self, *args, **kwargs) -> float:
            # Obtain the default reward, reflecting the increase in yield
            r = super()._get_reward(*args, **kwargs)
            # Balance the yield price with that of the costs of the applied N
            r = r * self._y_price - self._na * self._n_price
            # If N was applied, subtract the application costs
            if self._na != 0:
                r -= self._na_price
            return r


    from pcse.fileinput import CABOFileReader
    from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
    from pcse.util import WOFOST80SiteDataProvider

    env = CustomPCSEEnv(
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

    # Define an action that does nothing
    a = {
        'irrigation': 0,
        'N': 10,
        # 'P': 0,
        # 'K': 0,
    }

    o, r, done, info = env.step(a)

    # print(o)
    print(r)
