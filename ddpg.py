"""
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow

Original author: Patrick Emami

Author: Bart Keulen
"""

import numpy as np
import datetime
import gym
from gym.wrappers import Monitor
import tensorflow as tf
from actor import ActorNetwork
from critic import CriticNetwork
from replaybuffer import ReplayBuffer
from explorationnoise import ExplorationNoise

# ================================
#    TRAINING PARAMETERS
# ================================
# Learning rates actor and critic
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
# Maximum number of episodes
MAX_EPISODES = 1000
# Maximum number of steps per episode
MAX_STEPS_EPISODE = 500
# Discount factor
GAMMA = 0.99
# Soft target update parameter
TAU = 0.001
# Size of replay buffer
BUFFER_SIZE = 1000000
MINIBATCH_SIZE = 64
# Exploration noise variables
NOISE_MEAN = 0
NOISE_VAR = 1
# Ornstein-Uhlenbeck variables
OU_THETA = 0.15
OU_MU = 0.
OU_SIGMA = 0.3
# Exploration duration
EXPLORATION_TIME = 200


# ================================
#    UTILITY PARAMETERS
# ================================
# Gym environment name
ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'MountainCarContinuous-v0'
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = True
# Upload results to openAI
UPLOAD_GYM_RESULTS = False
GYM_API_KEY = '..............'
# Directory for storing gym results
DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
MONITOR_DIR = './results/{}/{}/gym_ddpg'.format(ENV_NAME, DATETIME)
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/{}/{}/tf_ddpg'.format(ENV_NAME, DATETIME)
RANDOM_SEED = 1234


# ================================
#    TENSORFLOW SUMMARY OPS
# ================================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar('Reward',  episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar('Qmax Value', episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ================================
#    TRAIN AGENT
# ================================
def train(sess, env, actor, critic):
    # Set up summary ops
    summary_ops, summary_vars = build_summaries()

    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in xrange(MAX_EPISODES):

        s = env.reset()

        episode_reward = 0
        episode_ave_max_q = 0

        noise = ExplorationNoise.ou_noise(OU_THETA, OU_MU, OU_SIGMA, MAX_STEPS_EPISODE)
        noise = ExplorationNoise.exp_decay(noise, EXPLORATION_TIME)

        for j in xrange(MAX_STEPS_EPISODE):

            if RENDER_ENV:
                env.render()

            # Add exploratory noise according to Ornstein-Uhlenbeck process to action
            # Decay exploration exponentially from 1 to 0 in EXPLORATION_TIME steps
            if i < EXPLORATION_TIME:
                a = actor.predict(np.reshape(s, (1, env.observation_space.shape[0]))) + noise[j]
            else:
                a = actor.predict(np.reshape(s, (1, env.observation_space.shape[0])))

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, actor.state_dim),
                              np.reshape(a, actor.action_dim), r, terminal,
                              np.reshape(s2, actor.state_dim))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    # If state is terminal assign reward only
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    # Else assgin reward + net target Q
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = \
                    critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                episode_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                a_grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, a_grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            episode_reward += r

            if terminal or j == MAX_STEPS_EPISODE-1:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: episode_reward,
                    summary_vars[1]: episode_ave_max_q
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print 'Reward: %.2i' % int(episode_reward), ' | Episode', i, \
                      '| Qmax: %.4f' % (episode_ave_max_q / float(j))

                break


# ================================
#    MAIN
# ================================
def main(_):
    with tf.Session() as sess:

        env = gym.make(ENV_NAME)
        # np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert(env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim, action_bound,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = Monitor(env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, actor, critic)

        # if UPLOAD_GYM_RESULTS:
            #gym.upload(MONITOR_DIR, api_key=GYM_API_KEY)

if __name__ == '__main__':
    tf.app.run()