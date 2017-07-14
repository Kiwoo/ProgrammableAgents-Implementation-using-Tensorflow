"""
Data structure for implementing actor network for DDPG algorithm
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Original author: Patrick Emami

Author: Bart Keulen
"""

import tensorflow as tf
import tflearn

class DetectorNetwork(object):

    def __init__(self, sess, properties, features_dim, learning_rate, tau):
        self.sess = sess
        self.features_dim = features_dim
        self.learning_rate = learning_rate
        self.tau = tau

        self.properties = [[1,2,3],[4, 5, 6],[7,8]] 
        # It should actually be
        # self.properties = properties


        # for example, if 1, 2, 3 are color related properties such as red, blue, green,
        # then we have to softmax the result of detector network to make it have mutually exclusive outputs.


        # Detector network

        self.inputs, self.outputs = self.create_detector_network()


        self.net_params = tf.trainable_variables()

        self.num_trainable_vars = len(self.net_params)

    def create_detector_network(self):
        inputs = tflearn.input_data(shape=[None, self.features_dim])

        for i in range(len(self.properties)):
            net = tflearn.fully_connected(inputs, 20, activation='relu')
            net = tflearn.fully_connected(net, 20, activation='relu')
        # Final layer weight are initialized to Uniform[-3e-3, 3e-3]
            weight_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            prpt_outputs = tflearn.fully_connected(net, len(self.properties[i]), activation='tanh', weights_init=weight_init)
            prpt_outputs = tf.add(tf.multiply(outputs, 0.5), 0.5) # Scale output to [-action_bound, action_bound]

        return inputs, outputs

    def mu_ex_network(self):

        # TODO:
        # construct mutually exclusive network for properties using softmax

    # def train(self, inputs, action_gradients):
    #     return self.sess.run(self.optimize, feed_dict={
    #         self.inputs: inputs,
    #         self.action_gradients: action_gradients
    #     })

    def detect(self, inputs):
        return self.sess.run(self.scaled_outputs, feed_dict={
            self.inputs: inputs
        })

    # def predict_target(self, inputs):
    #     return self.sess.run(self.target_scaled_outputs, feed_dict={
    #         self.target_inputs: inputs
    #     })

    # def update_target_network(self):
    #     self.sess.run(self.update_target_net_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
