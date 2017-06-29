import gym
import six
import numpy as np
from mujoco_utils import *

r = np.array(['0.1', '0.9', '0.1', '1.0'])
# g = [0.1, 0.9, 0.1, 1.0]
# b = [0.1, 0.1, 0.9, 1.0]

# print type(r)

env = gym.make('Reacher-v1')
env.reset()
# geom_rgba = env.env.model.geom_rgba
# print type(geom_rgba)
# # geom_rgba = np.array(geom_rgba)
# print type(geom_rgba)
# # print test_color
# geom_rgba = np.array(geom_rgba)
# geom_rgba[-3] = r
# print geom_rgba

# env.env.model.geom_rgba = geom_rgba
# print geom_rgba[-3][1]
# geom_rgba[-3] = 
# print test_color[-3]
# print test_color[-2]
# print test_color[-1]


# test_color[-3] = g
# test_color[-2] = g
# test_color[-1] = g

# model.geom_rgba = geom_rgba
# print type(model.geom_rgba)
# print env.env.model.geom_rgba

env.reset()
for _ in range(200):
    env.render()
    
    #body_pos = model.body_pos
    # target_name = 'target'
    # idx = model.body_names.index(six.b(target_name))
    # target_pos = get_body_pos(env, target_name)
    # print target_pos
    env.step(env.action_space.sample()) # take a random actio

print "End this episode"
env.close()
env = gym.make('Reacher-v2')
env.reset()
for _ in range(1000):
    env.render()
    
    #body_pos = model.body_pos
    # target_name = 'target'
    # idx = model.body_names.index(six.b(target_name))
    # target_pos = get_body_pos(env, target_name)
    # print target_pos
    env.step(env.action_space.sample()) # take a random actio
