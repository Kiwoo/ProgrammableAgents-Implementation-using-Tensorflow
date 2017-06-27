import numpy as np
import gym
import tensorflow as tf
import pickle
import logz
import scipy.signal
from tf_util import *



def main(): 
    env = gym.make("Reacher-v1")

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

 

if __name__ == "__main__":
    main()

