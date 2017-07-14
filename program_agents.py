import numpy as numpy



def get_featmap():
	feture_map = 0
	return feature_map

def feature_detector():
	feat_reprentation = 0
	return feat_reprentation

def interpret_command(in_command):
	return in_command

def apply_command(command):



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

        # TODO:
        # set num_objects, num_properties, features_dim for Detector network

        detector = DetectorNetwork(sess, num_objects, num_properties, features_dim,
                             ACTOR_LEARNING_RATE, TAU)


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