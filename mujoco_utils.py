import gym
import six

def get_body_pos(env, name_body):
    model = env.env.model
    idx = model.body_names.index(six.b(name_body))
    return model.data.com_subtree[idx]
