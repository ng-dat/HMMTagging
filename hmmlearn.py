import numpy as np
import sys
from collections import Counter

from model import HMM
import constant


def main(train_file_path):
    # Counting
    words = Counter()
    tags = Counter()
    tag_given_tag = Counter()
    word_given_tag = Counter()
    count = 0
    with open(train_file_path, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            tag_line = []
            pairs = line.split(' ')

            for pair in pairs:
                pair = pair.strip()
                if '/' not in pair:
                    continue
                    # TODO: check any case of this
                count += 1
                sep_pos = len(pair) - pair[::-1].index('/') - 1
                word = pair[:sep_pos]
                tag = pair[sep_pos+1:]
                words[word] += 1
                tags[tag] += 1
                word_given_tag[word+constant.GIVEN+tag] += 1
                tag_line.append(tag)
            if not tag_line:
                # print( "Empty")
                continue
            for i in range(len(tag_line)-1):
                tag_given_tag[tag_line[i+1]+constant.GIVEN+tag_line[i]] += 1
            tag_given_tag[constant.TAG_END+constant.GIVEN+tag_line[-1]] += 1

    # Get probabilities
    # for x in words:
    #     words[x] = words[x]/count #TODO: check if can using this
    for x in tags:
        tags[x] = tags[x]/count

    emission_prob = dict()
    for word in words.keys():
        emission_prob[word] = dict()
        for tag in tags.keys():
            emission_prob[word][tag] = word_given_tag[word + constant.GIVEN + tag] # TODO: smoothing
    for word in emission_prob.keys():
        total = sum(list(emission_prob[word].values()))
        for tag in tags.keys():
            emission_prob[word][tag] /= total

    transition_prob = dict()
    for tag in list(tags.keys())+[constant.TAG_END]:
        if tag not in transition_prob:
            transition_prob[tag] = dict()
        for prev_tag in tags.keys():
            transition_prob[tag][prev_tag] = max(1, tag_given_tag[tag + constant.GIVEN + prev_tag]) # TODO: smoothing
    for tag in list(tags.keys())+[constant.TAG_END]:
        total = sum(list(transition_prob[tag].values()))
        for prev_tag in tags.keys():
            transition_prob[tag][prev_tag] /= total

    count_vocab = Counter()
    for tag in tags.keys():
        count_related_words = 0
        for word in words.keys():
            if emission_prob[word][tag] > 0: count_related_words += 1
        count_vocab[tag] = count_related_words
    common_tags = [x[0] for x in count_vocab.most_common(int(len(tags)/2))]

    # Setting up model
    model = HMM(prior_prob=tags, trans_prob=transition_prob, emiss_prob=emission_prob,
                state_dict=set(tags.keys()), obs_dict=set(words.keys()), open_class_dict=common_tags)
    model.save_model()


if __name__ == '__main__':
    train_file_path = sys.argv[1]
    main(train_file_path)