import numpy as np
import gym
import tensorflow as tf
import pickle
import logz
import scipy.signal
from tf_util import *

'''
# moved to tf_util
def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)
'''

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary
'''
# moved to tf_util
def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), axis=1)
'''
def pathlength(path):
    return len(path["reward"])

class LinearValueFunction(object):
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

class NnValueFunction(object):

    def __init__(self):
        self.net = None

    def create_net(self, shape):
        print "Creat Net"
        self.x = tf.placeholder(shape=[None, shape], name="x", dtype=tf.float32)
        self.y = tf.placeholder(shape=[None], name="y", dtype=tf.float32)

        out = layers.fully_connected(self.x, num_outputs=5, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
        out = layers.fully_connected(out, num_outputs=3, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
        self.net = layers.fully_connected(out, num_outputs=1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        self.net = tf.reshape(self.net, (-1, ))
        l2 = (self.net - self.y) * (self.net - self.y)
        self.train = tf.train.AdamOptimizer(1e-4).minimize(l2)
        tf.global_variables_initializer().run()

    def _features(self, path):
        o = path["observation"]
        o = path["observation"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = len(path["reward"])
        ret = np.concatenate([o, o**2, np.ones((l, 1))], axis=1)
        return ret

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape[1])
        returns = np.concatenate([path["returns"] for path in paths])
        for _ in range(1000):
            get_session().run(self.train, {self.x: featmat, self.y: returns})

    def predict(self, path):
        if self.net is None:
            return np.zeros(len(path["reward"])) 
        else:
            ret = get_session().run(self.net, {self.x: self._features(path)})
            return np.reshape(ret, (ret.shape[0], ))

def main_cartpole(logdir, seed, n_iter, gamma, min_timesteps_per_batch, initial_stepsize, desired_kl, vf_type, vf_params, animate=False):
    env = gym.make("CartPole-v0")
    ob_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    logz.configure_output_dir(logdir)
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(**vf_params)

    stepsize = 1e-3
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate
    sy_h1 = lrelu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0), scope="H1")) # hidden layer
    sy_logits_na = dense(sy_h1, num_actions, "final", weight_init=normc_initializer(0.05), scope="FINAL") # "logits", describing probability distribution of final layer
    # we use a small initialization for the last layer, so the initial policy has maximal entropy
    sy_oldlogits_na = tf.placeholder(shape=[None, num_actions], name='oldlogits', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)
    sy_logp_na = tf.nn.log_softmax(sy_logits_na) # logprobability of actions
    sy_sampled_ac = categorical_sample_logits(sy_logits_na)[0] # sampled actions, used for defining the policy (NOT computing the policy gradient)
    sy_n = tf.shape(sy_ob_no)[0]
    sy_logprob_n = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n) # log-prob of actions taken -- used for policy gradient calculation

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
    sy_oldlogp_na = tf.nn.log_softmax(sy_oldlogits_na)
    sy_oldp_na = tf.exp(sy_oldlogp_na) 
    sy_kl = tf.reduce_sum(sy_oldp_na * (sy_oldlogp_na - sy_logp_na)) / tf.to_float(sy_n)
    sy_p_na = tf.exp(sy_logp_na)
    sy_ent = tf.reduce_sum( - sy_p_na * sy_logp_na) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    # use single thread. on such a small problem, multithreading gives you a slowdown
    # this way, we can better use multiple cores for different experiments
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    total_timesteps = 0

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            path["returns"] = return_t
            if vf_type == 'linear':
                vpred_t = vf.predict(path["observation"])
            elif vf_type == 'nn':
                vpred_t = vf.predict(path)
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        if vf_type == 'linear':
            vf.fit(ob_no, vtarg_n)
        elif vf_type == 'nn':
            vf.fit(paths)

        # Policy update
        _, oldlogits_na = sess.run([update_op, sy_logits_na], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldlogits_na:oldlogits_na})
        if kl > desired_kl * 2: 
            stepsize /= 1.5
            print('stepsize -> %s'%stepsize)
        elif kl < desired_kl / 2: 
            stepsize *= 1.5
            print('stepsize -> %s'%stepsize)
        else:
            print('stepsize OK')
        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        if vf_type == 'linear':
            logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        elif vf_type == 'nn':
            v_predicts = []
            for path in paths:
                v_predicts.append(np.vstack(vf.predict(path)))
            logz.log_tabular("EVAfter", explained_variance_1d(np.squeeze(np.vstack(np.array(v_predicts))), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

def main_pendulum(logdir, seed, n_iter, gamma, min_timesteps_per_batch, initial_stepsize, desired_kl, vf_type, vf_params, animate=False):
    # n_iter=3000, gamma=0.97, min_timesteps_per_batch=10000, stepsize=1e-3, animate=True, logdir=None
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    num_actions = np.prod(env.action_space.shape)
    logz.configure_output_dir(logdir)
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(**vf_params)

    stepsize = 1e-3

    config = dict2(**{
        "max_kl": 0.01,
        "cg_damping": 0.01,
        "gamma": 0.97,
        "animation" : 20})

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None, num_actions], name="ac", dtype=tf.float32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate


    sy_h1 = tf.nn.tanh(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0), scope="H1")) # hidden layer
    sy_logits_mu_na = dense(sy_h1, num_actions, "final", weight_init=normc_initializer(0.05), scope="FINAL") # "logits", describing probability distribution of final layer


    sy_logits_logstd_param = tf.Variable((.01*np.random.randn(1, num_actions)).astype(np.float32))
    sy_logits_logstd_na = tf.tile(sy_logits_logstd_param, tf.stack((tf.shape(sy_logits_mu_na)[0], 1)))

    
    # we use a small initialization for the last layer, so the initial policy has maximal entropy
    sy_oldlogits_mu_na = tf.placeholder(shape=[None, num_actions], name='oldlogits_mu', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)
    sy_oldlogits_logstd_na = tf.placeholder(shape=[None, num_actions], name='oldlogits_std', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)

    sy_logp_na = gauss_log_prob(sy_logits_mu_na, sy_logits_logstd_na, sy_ac_n)
    sy_oldlogp_na = gauss_log_prob(sy_oldlogits_mu_na, sy_oldlogits_logstd_na, sy_ac_n)

    ratio_n = tf.exp(sy_logp_na - sy_oldlogp_na)

    sy_n = tf.shape(sy_ob_no)[0]

    sy_surr = - tf.reduce_mean(ratio_n * sy_adv_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_kl = gauss_KL(sy_oldlogits_mu_na, sy_oldlogits_logstd_na,
                      sy_logits_mu_na, sy_logits_logstd_na) / tf.to_float(sy_n)

    sy_ent = gauss_ent(sy_logits_mu_na, sy_logits_logstd_na) / tf.to_float(sy_n)


    losses = [sy_surr, sy_kl, sy_ent]

    eps = 1e-8

    kl_firstfixed = gauss_selfKL_firstfixed(sy_logits_mu_na, sy_logits_logstd_na) / tf.to_float(sy_n)

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    sess = tf.Session()
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    total_timesteps = 0

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        # print "c2"
        while True:
            ob = env.reset()
            # print "c3"
            terminated = False
            obs, acs, rewards, action_dists_mu, action_dists_logstd = [], [], [], [], []
            animate_this_episode=(len(paths)==0 and (i % config.animation == 0) and animate)
            while True:
                obs.append(ob)
                ac_mu, ac_log_std = sess.run([sy_logits_mu_na, sy_logits_logstd_na], feed_dict={sy_ob_no : ob[None]})
                ac = ac_mu + np.exp(ac_log_std)*np.random.randn(*ac_log_std.shape)
                ac= ac.ravel()
                acs.append(ac)
                action_dists_mu.append(ac_mu)
                action_dists_logstd.append(ac_log_std)

                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), 
                    "terminated" : terminated,
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs), 
                    "action_dists_mu": np.concatenate(action_dists_mu),
                    "action_dists_logstd": np.concatenate(action_dists_logstd)}

            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        vtargs, vpreds, advs = [], [], []
        
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            path["returns"] = return_t
            if vf_type == 'linear':
                vpred_t = vf.predict(path["observation"])
            elif vf_type == 'nn':
                vpred_t = vf.predict(path)
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)



        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        ac_dist_mu_n = np.concatenate([path["action_dists_mu"] for path in paths])
        ac_dist_logstd_n = np.concatenate([path["action_dists_logstd"] for path in paths])
        adv_n = np.concatenate(advs)

        
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)

        if vf_type == 'linear':
            vf.fit(ob_no, vtarg_n)
        elif vf_type == 'nn':
            vf.fit(paths)


        sess.run(update_op, feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_oldlogits_mu_na: ac_dist_mu_n, sy_oldlogits_logstd_na: ac_dist_logstd_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldlogits_mu_na: ac_dist_mu_n, sy_oldlogits_logstd_na: ac_dist_logstd_n})

        if kl > desired_kl * 2: 
            stepsize /= 1.5
            print('stepsize -> %s'%stepsize)
        elif kl < desired_kl / 2: 
            stepsize *= 1.5
            print('stepsize -> %s'%stepsize)
        else:
            print('stepsize OK')


        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        if vf_type == 'linear':
            logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        elif vf_type == 'nn':
            v_predicts = []
            for path in paths:
                v_predicts.append(np.vstack(vf.predict(path)))
            logz.log_tabular("EVAfter", explained_variance_1d(np.squeeze(np.vstack(np.array(v_predicts))), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()


def trpo_hopper(logdir, n_iter=501, min_timesteps_per_batch=10000, stepsize=1e-2, animate=True): #gamma=0.99, 
    env = gym.make("Hopper-v1")

    ob_dim = env.observation_space.shape[0]
    num_actions = np.prod(env.action_space.shape)

    logz.configure_output_dir(logdir)

    config = dict2(**{
        "max_kl": 0.01,
        "cg_damping": 0.01,
        "gamma": 0.995,
        "animation" : 20})

    print "{}, {}".format(ob_dim, num_actions)

    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None, num_actions], name="ac", dtype=tf.float32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate


    sy_h1 = tf.nn.tanh(dense(sy_ob_no, 64, "h1", weight_init=normc_initializer(1.0), scope="H1")) # hidden layer
    sy_h2 = tf.nn.tanh(dense(sy_h1, 64, "h2", weight_init=normc_initializer(1.0), scope="H2")) # hidden layer
    sy_logits_mu_na = dense(sy_h2, num_actions, "final", weight_init=normc_initializer(0.05), scope="FINAL") # "logits", describing probability distribution of final layer


    sy_logits_logstd_param = tf.Variable((.01*np.random.randn(1, num_actions)).astype(np.float32))
    sy_logits_logstd_na = tf.tile(sy_logits_logstd_param, tf.stack((tf.shape(sy_logits_mu_na)[0], 1)))

    sy_oldlogits_mu_na = tf.placeholder(shape=[None, num_actions], name='oldlogits_mu', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)
    sy_oldlogits_logstd_na = tf.placeholder(shape=[None, num_actions], name='oldlogits_std', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)

    sy_logp_na = gauss_log_prob(sy_logits_mu_na, sy_logits_logstd_na, sy_ac_n)
    sy_oldlogp_na = gauss_log_prob(sy_oldlogits_mu_na, sy_oldlogits_logstd_na, sy_ac_n)

    ratio_n = tf.exp(sy_logp_na - sy_oldlogp_na)

    sy_n = tf.shape(sy_ob_no)[0]

    sy_surr = - tf.reduce_mean(ratio_n * sy_adv_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_kl = gauss_KL(sy_oldlogits_mu_na, sy_oldlogits_logstd_na,
                      sy_logits_mu_na, sy_logits_logstd_na) / tf.to_float(sy_n)

    sy_ent = gauss_ent(sy_logits_mu_na, sy_logits_logstd_na) / tf.to_float(sy_n)


    losses = [sy_surr, sy_kl, sy_ent]
    var_list = tf.trainable_variables()

    eps = 1e-8
    pg = flatgrad(sy_surr, var_list)

    kl_firstfixed = gauss_selfKL_firstfixed(sy_logits_mu_na, sy_logits_logstd_na) / tf.to_float(sy_n)

    grads = tf.gradients(kl_firstfixed, var_list)

    flat_tangent = tf.placeholder(shape=[None], dtype=tf.float32)
    shapes = map(var_shape, var_list)


    start = 0
    tangents = []

    for shape in shapes:
        size = np.prod(shape)
        param = tf.reshape(flat_tangent[start:(start + size)], shape)
        tangents.append(param)
        start += size

    gvp     = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
    fvp     = flatgrad(gvp, var_list)
    gf      = GetFlat(var_list)
    sff     = SetFromFlat(var_list) 
    vf_1      = LinearValueFunction()

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(stepsize).minimize(sy_surr)

    sess = tf.Session()
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    total_timesteps = 0
    cur_division = 1

    max_kl = config.max_kl
    pickle_save = 10

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards, action_dists_mu, action_dists_logstd = [], [], [], [], []
            animate_this_episode=(len(paths)==0 and (i % config.animation == 0) and animate)
            while True:
                obs.append(ob)
                ac_mu, ac_log_std = sess.run([sy_logits_mu_na, sy_logits_logstd_na], feed_dict={sy_ob_no : ob[None]})
                ac = ac_mu + np.exp(ac_log_std)*np.random.randn(*ac_log_std.shape)
                ac= ac.ravel()
                acs.append(ac)
                action_dists_mu.append(ac_mu)
                action_dists_logstd.append(ac_log_std)

                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), 
                    "terminated" : terminated,
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs), 
                    "action_dists_mu": np.concatenate(action_dists_mu),
                    "action_dists_logstd": np.concatenate(action_dists_logstd)}

            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break

        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, config.gamma)
            path["returns"] = return_t 
            vpred_t = vf_1.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        ac_dist_mu_n = np.concatenate([path["action_dists_mu"] for path in paths])
        ac_dist_logstd_n = np.concatenate([path["action_dists_logstd"] for path in paths])
        adv_n = np.concatenate(advs)

        
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)

        vf_1.fit(ob_no, vtarg_n)
        thprev = gf()

        feed = {sy_ob_no:ob_no, 
                sy_ac_n:ac_n, 
                sy_adv_n: standardized_adv_n, 
                sy_oldlogits_mu_na: ac_dist_mu_n,
                sy_oldlogits_logstd_na: ac_dist_logstd_n}

        def fisher_vector_product(p):
            feed[flat_tangent] = p
            return sess.run(fvp, feed) + config.cg_damping * p

        g = sess.run(pg, feed_dict = feed)
        stepdir = conjugate_gradient(fisher_vector_product, -g)

        divide_iteration = 20

        n = n_iter / divide_iteration
        n = n * cur_division
        if i > n:            
            max_kl = config.max_kl - config.max_kl * cur_division / divide_iteration
            cur_division = cur_division + 1   
        if i%10==0:
            print "cur max_kl : {}".format(max_kl)

        shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
        lm = np.sqrt(shs / max_kl)
        fullstep = stepdir / lm
        neggdotstepdir = -g.dot(stepdir)

        def loss(th):
            sff(th)
            return sess.run(losses[0], feed_dict=feed)

        theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
        sff(theta)

        surrafter, kloldnew, entropy = sess.run(
            losses, feed_dict=feed)

        if kloldnew > 2.0 * max_kl:
            print "not updated"
            sff(thprev)

        if i%pickle_save == 0:
            H1_vars = scope_vars(scope="H1", trainable_only=True)
            for var in H1_vars:
                if var.name == "H1/h1/w:0":
                    W1 = var.eval()
                elif var.name == "H1/h1/b:0":
                    b1 = var.eval()

            H2_vars = scope_vars(scope="H2", trainable_only=True)
            for var in H2_vars:
                if var.name == "H2/h2/w:0":
                    W2 = var.eval()
                elif var.name == "H2/h2/b:0":
                    b2 = var.eval()

            FINAL_vars = scope_vars(scope="FINAL", trainable_only=True)
            for var in FINAL_vars:
                if var.name == "FINAL/final/w:0":
                    W3 = var.eval()
                elif var.name == "FINAL/final/b:0":
                    b3 = var.eval()                       
            data = {'W1'    : W1,
                    'b1'    : b1,
                    'W2'    : W2,
                    'b2'    : b2,
                    'W3'    : W3,
                    'b3'    : b3,
                    }
            file_name = "expert_11th_{}.pkl".format(i)     
            f = open(file_name, 'wb')
            pickle.dump(data, f)   

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpBest100RewMean", np.amax([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kloldnew)
        logz.log_tabular("Entropy", entropy)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf_1.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

def main_pendulum1(d):
    return main_pendulum(**d)

def main_cartpole1(d):
    return main_cartpole(**d)    

if __name__ == "__main__":
    # trpo_hopper(logdir='trpo')

    '''
    If you want to run trpo-hopper, please uncommnet above line "trpo_hopper(logdir='trpo')" and comment out below lines
    '''
    if 0:
        main_pendulum(logdir=None) # when you want to start collecting results, set the logdir
    if 1:
        general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=100, initial_stepsize=1e-3)
        # params = [
        #     dict(logdir='/tmp/ref/linearvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
        #     dict(logdir='/tmp/ref/nnvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
        #     dict(logdir='/tmp/ref/linearvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
        #     dict(logdir='/tmp/ref/nnvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
        #     dict(logdir='/tmp/ref/linearvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
        #     dict(logdir='/tmp/ref/nnvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
        # ]
        params = dict(logdir='11th', seed=0, desired_kl=2e-3, vf_type='nn', vf_params={}, **general_params)
        # main_pendulum1(params)
        main_pendulum1(params)

