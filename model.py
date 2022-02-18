import numpy as np
import json

import constant


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
        tags = list(self.state_dict)
        input_len = len(obs_seq)
        tag_num = len(tags)
        delta = np.zeros([tag_num, input_len])
        Delta = np.zeros([tag_num, input_len])
        NEG_INFI = np.log(1e-9)#np.float('-inf')
        for t in range(tag_num):
            if self.prior_prob[tags[t]] == 0 or obs_seq[0] not in self.obs_dict or self.emiss_prob[obs_seq[0]][tags[t]] == 0:
                delta[t][0] = NEG_INFI # TODO: smoothing
            else:
                delta[t][0] = np.log(self.prior_prob[tags[t]]) + np.log(self.emiss_prob[obs_seq[0]][tags[t]])
        for i in range(1, input_len):
            emiss_prob = NEG_INFI  # TODO: handle unknown words
            for t in range(tag_num):
                if obs_seq[i] in self.obs_dict and self.emiss_prob[obs_seq[i]][tags[t]] != 0:
                    emiss_prob = np.log(self.emiss_prob[obs_seq[i]][tags[t]])
                delta[t][i] = np.float('-inf')
                for prev_t in range(tag_num):
                    # if delta[prev_t][i-1] < 0:
                    #     continue # TODO: handle this case (when?)
                    # if self.trans_prob[tags[t]][tags[prev_t]] == 0:
                    #     cur_prob = np.log(1e-9) + delta[prev_t][i-1] # TODO: handle non-exist transition
                    # else:
                    #     cur_prob = np.log(self.trans_prob[tags[t]][tags[prev_t]]) + delta[prev_t][i-1]
                    cur_prob = np.log(self.trans_prob[tags[t]][tags[prev_t]]) + delta[prev_t][i - 1]
                    if delta[t][i] <= cur_prob:
                            delta[t][i] = cur_prob
                            Delta[t][i] = prev_t
                delta[t][i] += emiss_prob

        most_likely_path = [np.argmax([delta[last_tag][input_len-1] + \
                                       np.log(self.trans_prob[constant.TAG_END][tags[last_tag]]) for last_tag in range(tag_num)])]
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

