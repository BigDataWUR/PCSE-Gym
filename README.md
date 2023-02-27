## Introduction
CropGym is a highly configurable [Python Open AI gym](https://gym.openai.com/) environment to conduct Reinforcement Learning (RL) research for crop management. CropGym is built around [PCSE](https://pcse.readthedocs.io/en/stable/), a well established python library that includes implementations of a variety of crop simulation models. CropGym follows standard gym conventions and enables daily interactions between an RL agent and a crop model.

## Code 
Source code is available at [https://github.com/BigDataWUR/PCSE-Gym](https://github.com/BigDataWUR/PCSE-Gym).

## Installation instructions
To install a minimalistic version, do the following:

1. Clone [PCSE](https://github.com/ajwdewit/pcse.git)
2. Clone [CropGym](https://github.com/BigDataWUR/PCSE-Gym)

The code has been tested using python 3.8.10.
## Example
- [Basic example](tutorials/basic.md)
- [Advanced example](tutorials/customization.md)

## Use case
A use case is built for nitrogen fertilization in winter wheat. A Jupiter notebook (readily usable in [Google Colab](https://colab.research.google.com/)) showing a trained RL agent in action can be found [here](notebooks/nitrogen-winterwheat/results_paper.ipynb).
