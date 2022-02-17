import numpy as np


class HMM:
    def __init__(self, prior_prob, trans_prob, obs_prob, state_dict, obs_dict):
        """
        :param prior_prob: prior probabilities. Aka P(tag)
        :param trans_prob: transition probabilities. Aka P(tag1 | tag2)
        :param obs_prob: emission probabilities. Aka P(word | tag)
        :param state_dict: dictionary of all states. Aka tags
        :param obs_dict: dictionary of all observations. Aka words
        """
        self.prior_prob = prior_prob
        self.trans_prob = trans_prob
        self.obs_prob = obs_prob
        self.state_dict = state_dict
        self.obs_dict = obs_dict

    def forward(self, obs_seq):
        pass

    def backward(self, obs_seq):
        pass

    def viterbi(self, obs_seq):
        """
            Input an observation sequences, return the most likely sequence of hidden states
        """
        pass
