#
# if __name__ == '__main__':
#
#     from collections import namedtuple, deque
#     import random
#     import math
#     from itertools import count
#
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     import torch.nn.functional as F
#     import torchvision.transforms as T
#
#     import pcse
#     import pcse_gym.environment.env
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
#     """
#
#         Demo of using Reinforcement Learning in the PCSE-gym
#
#         Most of the code/setup has been copied from the PyTorch Reinforcement Learning tutorial
#         https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#
#     """
#
#
#     """
#     Define environment
#     """
#
#     class Env(pcse_gym.environment.env.PCSEEnv):
#
#         def _get_reward(self, _) -> float:
#             output = self._model.get_output()
#
#             var, scale = 'TGROWTH', 1e-2
#             # var, scale = 'TNSOIL', 1
#
#             # Consider different cases:
#             if len(output) == 0:
#                 return 0
#             if len(output) <= self._timestep:
#                 return output[-1][var] * scale
#             else:
#                 return (output[-1][var] - output[-self._timestep - 1][var]) * scale
#
#         def _apply_action(self, action):
#             self._model._send_signal(signal=pcse.signals.apply_n,
#                                      amount=action['N'],
#                                      recovery=0.7
#                                      )
#
#
#     """
#         Define some helper classes
#     """
#
#     Transition = namedtuple('Transition',
#                             ('state', 'action', 'next_state', 'reward'))
#
#
#     class ReplayMemory(object):
#
#         def __init__(self, capacity):
#             self.memory = deque([], maxlen=capacity)
#
#         def push(self, *args):
#             """Save a transition"""
#             self.memory.append(Transition(*args))
#
#         def sample(self, batch_size):
#             return random.sample(self.memory, batch_size)
#
#         def __len__(self):
#             return len(self.memory)
#
#     """
#         Define a Deep Q-Network
#     """
#
#
#     class DQN(nn.Module):
#
#         def __init__(self, num_days, input_size, outputs):
#             super(DQN, self).__init__()
#             self.l1 = nn.Linear(input_size, 64)
#             self.l2 = nn.Linear(64, outputs)
#
#         def forward(self, x):
#             x = x.to(device)
#             x = F.relu(self.l1(x))
#             return self.l2(x)
#
#
#     BATCH_SIZE = 128
#     GAMMA = 0.999
#     EPS_START = 0.9
#     EPS_END = 0.05
#     EPS_DECAY = 200
#     TARGET_UPDATE = 10
#
#     timestep = 7  # days
#     env = Env(timestep=timestep)
#
#     # Get number of actions from gym action space
#     # action_space = [0, 1, 10, 100, 1000, 10000, 100000]  # TODO -- right values
#     action_space = [0, 10]
#     # action_space = [0, 0, 0, 0, 0, 0]
#     n_actions = len(action_space)
#
#     state_variables = env.output_variables + env.weather_variables
#
#     policy_net = DQN(timestep, len(state_variables), n_actions).to(device)
#     target_net = DQN(timestep, len(state_variables), n_actions).to(device)
#     target_net.load_state_dict(policy_net.state_dict())
#     target_net.eval()
#
#     optimizer = optim.Adam(policy_net.parameters())
#     memory = ReplayMemory(10000)
#
#     steps_done = 0
#
#
#     def select_action(state):
#         global steps_done
#         sample = random.random()
#         eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#             math.exp(-1. * steps_done / EPS_DECAY)
#         steps_done += 1
#         if sample > eps_threshold:
#             with torch.no_grad():
#                 # t.max(1) will return largest column value of each row.
#                 # second column on max result is index of where max element was
#                 # found, so we pick action with the larger expected reward.
#                 return policy_net(state).max(1)[1].view(1, 1)
#         else:
#             return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
#
#
#     episode_durations = []
#
#
#     def optimize_model():
#         if len(memory) < BATCH_SIZE:
#             return
#         transitions = memory.sample(BATCH_SIZE)
#         # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#         # detailed explanation). This converts batch-array of Transitions
#         # to Transition of batch-arrays.
#         batch = Transition(*zip(*transitions))
#
#         # Compute a mask of non-final states and concatenate the batch elements
#         # (a final state would've been the one after which simulation ended)
#         non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                               batch.next_state)), device=device, dtype=torch.bool)
#         non_final_next_states = torch.cat([s for s in batch.next_state
#                                                     if s is not None])
#         state_batch = torch.cat(batch.state)
#         action_batch = torch.cat(batch.action)
#         reward_batch = torch.cat(batch.reward)
#
#         # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#         # columns of actions taken. These are the actions which would've been taken
#         # for each batch state according to policy_net
#         state_action_values = policy_net(state_batch).gather(1, action_batch)
#
#         # Compute V(s_{t+1}) for all next states.
#         # Expected values of actions for non_final_next_states are computed based
#         # on the "older" target_net; selecting their best reward with max(1)[0].
#         # This is merged based on the mask, such that we'll have either the expected
#         # state value or 0 in case the state was final.
#         next_state_values = torch.zeros(BATCH_SIZE, device=device)
#         next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
#         # Compute the expected Q values
#         expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#         # Compute Huber loss
#         criterion = nn.SmoothL1Loss()
#         loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
#
#         # Optimize the model
#         optimizer.zero_grad()
#         loss.backward()
#         for param in policy_net.parameters():
#             param.grad.data.clamp_(-1, 1)
#         optimizer.step()
#
#
#     def action_to_dict(a):
#         return {
#             'irrigation': 0,
#             'P': 0,
#             'K': 0,
#             'N': a,
#         }
#
#     def observation_from_dict(o):
#         o_c = o['crop_model']
#         o_w = o['weather']
#         o = {**o_c, **o_w}
#         o = [sum(o[v]) / len(o[v]) for v in state_variables]
#         o = torch.tensor(o).view(1, -1)
#         o = normalize_observation(o)
#         # print(o)
#         return o
#
#
#     def normalize_observation(o):
#         means = torch.tensor([0.6135388459324724, 0.026552096356100983, 0.004880498672844524, 0.023805761547334627, 0.06320754716981102, -0.0332075471698113, 8.268107949289973, 0.8491430075385181, 781.05703388043, 1.4645126394470565, 7.9621108091842885, 0.9866543909912389, 0.0, 2.9328891908156667, 285.09179245283025, 0.9920226093968921, 0.0, 691.9352357776855, 90.2554716463884, 0.3061224434912368, 79.14073214191492, 60.62226520796884, 59.030557751243904, 146.82899847466746, 405.34323995313423, 12568915.094339622, 7.049103773584905, 15.64221698113208, 11.332763140672757, 0.279933962264151, 0.2367371491011578, 0.20654893059135546, 0.21659911962419887, 3.4223584905660367])
#         stds = torch.tensor([129.4572985342988, 5.602535312705799, 1.029787827675611, 5.023026928745189, 13.337265706397323, 7.007693206056765, 1744.5916540735627, 179.17040631939338, 164804.56492936795, 309.0159532624815, 1680.0137555355777, 208.18657252593368, 1.0, 618.8503676300702, 60154.55184962901, 209.3167745744679, 1.0, 145999.75387381812, 19044.091898500963, 64.59185018403895, 16698.945812343984, 12791.476595286296, 12455.548047854145, 30981.74009574724, 85528.26248712119, 2652051568.3018966, 1487.3687869537412, 3300.513606412874, 2391.215781113961, 59.06742123856081, 49.95181043640927, 43.58206577929977, 45.70259310915408, 722.1192757517717])
#         return (o - means) * 1000 / stds
#
#
#     """
#         Main loop
#     """
#
#     num_episodes = 500
#     for i_episode in range(num_episodes):
#         # Initialize the environment and state
#         state, _ = env.reset()
#         state = observation_from_dict(state)
#
#         rewards = []
#
#         for t in count():
#             # Select and perform an action
#             action_tensor = select_action(state)
#             action = action_to_dict(action_space[action_tensor.item()])
#             # action = action_to_dict(10)
#
#             next_state, reward, done, _ = env.step(action)
#             reward = reward
#             rewards += [reward]
#             # print(reward)
#             next_state = observation_from_dict(next_state)
#             reward = torch.tensor([reward], device=device)
#
#             # Store the transition in memory
#             memory.push(state, action_tensor, next_state, reward)
#
#             # Move to the next state
#             state = next_state
#
#             # Perform one step of the optimization (on the policy network)
#             optimize_model()
#             if done:
#                 episode_durations.append(t + 1)
#                 print(sum(rewards))
#                 break
#
#         # Update the target network, copying all weights and biases in DQN
#         if i_episode % TARGET_UPDATE == 0:
#             target_net.load_state_dict(policy_net.state_dict())
#
