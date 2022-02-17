import numpy as np
import json


class HMM:
    def __init__(self, prior_prob=None, trans_prob=None, emiss_prob=None, state_dict=None, obs_dict=None):
        """
        :param prior_prob: prior probabilities. Aka P(tag)
        :param trans_prob: transition probabilities. Aka P(tag | other_preious_tags)
        :param emiss_prob: emission probabilities. Aka P(word | tag)
        :param state_dict: dictionary of all states. Aka tags
        :param obs_dict: dictionary of all observations. Aka words
        """
        if prior_prob and trans_prob and emiss_prob and state_dict and obs_dict:
            self.prior_prob = prior_prob
            self.trans_prob = trans_prob
            self.emiss_prob = emiss_prob
            self.state_dict = state_dict
            self.obs_dict = obs_dict
        else:
            self.prior_prob = dict()
            self.trans_prob = dict()
            self.emiss_prob = dict()
            self.state_dict = set()
            self.obs_dict = set()

    def forward(self, obs_seq):
        pass

    def backward(self, obs_seq):
        pass

    def viterbi(self, obs_seq):
        """
            Input an observation sequences, return the most likely sequence of hidden states
        """
        pass

    def save_model(self):
        data = json.dumps({'prior': self.prior_prob,
                           'trans_prob': self.trans_prob,
                           'emiss_prob': self.emiss_prob,
                           'state_dict': list(self.state_dict),
                           'obs_dict': list(self.obs_dict)}, indent=1)
        with open('hmmmodel.txt', 'w') as model_file:
            model_file.write(data)
            model_file.close()

    def load_model(self):
        with open('hmmmodel.txt', 'r') as model_file:
            data = json.load(model_file)
            model_file.close()
        self.prior_prob, self.trans_prob, self.emiss_prob, self.state_dict, self.obs_dict = \
            data['prior'], data['trans_prob'], data['emiss_prob'], set(data['state_dict']), set(data['obs_dict'])

    def set_param(self):
        pass

