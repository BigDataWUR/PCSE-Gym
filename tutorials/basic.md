
Here we show a basic example of how to use the PCSE Environment

```python

# Import the PCSE Environment class
from pcse_gym.environment.env import PCSEEnv

# PCSE contains various utility classes to load parameter configurations
from pcse.fileinput import CABOFileReader
from pcse.util import WOFOST72SiteDataProvider

# Create and configure a PCSE-Gym environment
# Note: the following configuration has not been chosen for its realism
env = PCSEEnv(
    model_config='Wofost72_WLP_FD.conf',
    agro_config='../pcse_gym/environment/configs/agro/sugarbeet_calendar.yaml',
    crop_parameters=CABOFileReader('../pcse_gym/environment/configs/crop/SUG0601.CAB'),
    site_parameters=WOFOST72SiteDataProvider(WAV=10),
    soil_parameters=CABOFileReader('../pcse_gym/environment/configs/soil/ec3.CAB'),
)

# Reset/initialize the environment to obtain an initial observation
o = env.reset()

```

By default, the PCSE Environment observations contain the crop model output variables as specified by the config file (in this case Wofost72_WLP_FD.conf), as well as weather statistics. Printing the observation gives the following initial information:

```python
{
   "crop_model":{
      "DVS":[0.0],
      "LAI":[0.0006936],
      "TAGP":[0.40800000000000003],
      "TWSO":[0.0],
      "TWLV":[0.3468],
      "TWST":[0.061200000000000004],
      "TWRT":[0.10200000000000001],
      "TRA":[5.411819351515599e-05],
      "RD":[10.0],
      "SM":[0.4],
      "WWLOW":[22.479999999999997]
   },
   "weather":{
      "IRRAD":[16610000.0],
      "TMIN":[-1.63],
      "TMAX":[7.12],
      "VAP":[6.046046468551459],
      "RAIN":[0.022],
      "E0":[0.16351492813999075],
      "ES0":[0.13357184194043034],
      "ET0":[0.1508001566684669],
      "WIND":[2.65]
   }
}
```

Next, we can define actions to apply to the crops. By default, the PCSE gym supports irrigation and fertilization

```python

# Define an action that does nothing
a = {
    'irrigation': 0,
    'N': 0,
    'P': 0,
    'K': 0,
}

# Apply it to our environment, to see how the crops develop in 1 day without interference
o, r, done, info = env.step(a)

```
From the model, we obtain an observation of how the crops behave on day 2. Also, we obtain a scalar reward that indicates the desirability of the current crop state. By default, this has been set to the difference in WSO (weight storage organ, that is eventually the yield that is harvested) that was accumulated during this time step. Furthermore, the environment gives a boolean `done` flag indicating whether the environment has terminated, as well as an `info` dict that provides the possibility of returning additional information that might be of interest for analysis/debugging.

We can run the model until termination, to observe how the crops would develop completely without interference:

```python
r_sum = 0
while not done:
    o, r, done, info = env.step(a)
    r_sum += r
```

The main objective of reinforcement learning is to build a policy that dictates when to choose which actions to maximize the expected eventual sum of rewards.

