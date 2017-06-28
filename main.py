import gym
import six
from mujoco_utils import *

env = gym.make('Reacher-v1')
env.reset()
for _ in range(1000):
    env.render()
    model = env.env.model
    #body_pos = model.body_pos
    target_name = 'target'
    # idx = model.body_names.index(six.b(target_name))
    target_pos = get_body_pos(env, target_name)
    print target_pos
    env.step(env.action_space.sample()) # take a random actio