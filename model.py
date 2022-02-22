import numpy as np
import json

import constant


class HMM:
    def __init__(self, prior_prob=None, trans_prob=None, emiss_prob=None, state_dict=None, obs_dict=None, open_class_dict=None):
        """
        :param prior_prob: prior probabilities. Aka P(tag)
        :param trans_prob: transition probabilities. Aka P(tag | other_preious_tags)
        :param emiss_prob: emission probabilities. Aka P(word | tag)
        :param state_dict: dictionary of all states. Aka tags
        :param obs_dict: dictionary of all observations. Aka words
        :param open_class_dict: dictionary of popular states, used when tagging UNK observations. Popular states mean those with majority of related observations
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
        self.open_class_dict = set() if not open_class_dict else open_class_dict

    def forward(self, obs_seq):
        pass

    def backward(self, obs_seq):
        pass

    def viterbi(self, obs_seq):
        """
            Input an observation sequences, return the most likely sequence of hidden states
        """
        tags = list(self.state_dict)
        input_len = len(obs_seq)
        tag_num = len(tags)
        delta = np.zeros([tag_num, input_len])
        Delta = np.zeros([tag_num, input_len])

        for t in range(tag_num):
            if obs_seq[0] not in self.obs_dict:
                if len(self.open_class_dict) > 0 and tags[t] in self.open_class_dict:
                    delta[t][0] = self.prior_prob[tags[t]]
                else:
                    delta[t][0] = 0
            else:
                delta[t][0] = self.prior_prob[tags[t]] * self.emiss_prob[obs_seq[0]][tags[t]]
        for i in range(1, input_len):
            for t in range(tag_num):
                if obs_seq[i] in self.obs_dict and self.emiss_prob[obs_seq[i]][tags[t]] == 0:
                    delta[t][i] = 0
                    continue
                if obs_seq[i] not in self.obs_dict and tags[t] not in self.open_class_dict: # when Tag not in open-class
                    delta[t][i] = 0
                    continue

                emiss_prob = 1  # TODO: handle unknown words
                if obs_seq[i] in self.obs_dict:
                    emiss_prob = self.emiss_prob[obs_seq[i]][tags[t]]

                cur_prob = [self.trans_prob[tags[t]][tags[prev_t]] * delta[prev_t][i - 1] for prev_t in range(tag_num)]
                delta[t][i] = np.max(cur_prob)
                Delta[t][i] = np.argmax(cur_prob)

                delta[t][i] *= emiss_prob

        most_likely_path = [np.argmax([delta[last_tag][input_len-1] * \
                                       self.trans_prob[constant.TAG_END][tags[last_tag]] for last_tag in range(tag_num)])]
        for i in range(input_len-1,0,-1):
            most_likely_path.append(int(Delta[most_likely_path[-1]][i]))
        most_likely_path.reverse()

        output = [tags[t] for t in most_likely_path]
        return output

    def save_model(self):
        data = json.dumps({'prior': self.prior_prob,
                           'trans_prob': self.trans_prob,
                           'emiss_prob': self.emiss_prob,
                           'state_dict': list(self.state_dict),
                           'obs_dict': list(self.obs_dict),
                           'open_class': list(self.open_class_dict)}, indent=1)
        with open('hmmmodel.txt', 'w') as model_file:
            model_file.write(data)
            model_file.close()

    def load_model(self):
        with open('hmmmodel.txt', 'r') as model_file:
            data = json.load(model_file)
            model_file.close()
        self.prior_prob, self.trans_prob, self.emiss_prob, self.state_dict, self.obs_dict, self.open_class_dict = \
            data['prior'], data['trans_prob'], data['emiss_prob'], set(data['state_dict']), set(data['obs_dict']), set(data['open_class'])

    def set_param(self):
        pass

