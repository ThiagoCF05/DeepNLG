__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 26/03/2019
Description:
    This script aims to perform the Majority implementation of the Discourse Ordering step.

    ARGS:
        [1] Path to the textual file where the inputs are
        [2] Path to the textual file where the outputs should be saved
        [3] Path to train set

    EXAMPLE:
        python3 major.py ordering/dev.eval ordering/dev.major.out ordering/data/train.json
"""

import sys
sys.path.append('../')
sys.path.append('./')
import json
import operator
import os
import utils

from collections import Counter

class MajorOrder():
    def __init__(self, trainpath):
        traindata = json.load(open(trainpath))
        self.train(traindata)


    def train(self, data):
        self.model = {}
        for entry in data:
            src_triples = utils.split_triples(entry['source'])
            # source predicates
            src_preds = tuple(sorted([t[1] for t in src_triples]))
            if src_preds not in self.model:
                self.model[src_preds] = []

            for target in entry['targets']:
                output = utils.split_triples(target['output'])
                # target predicates
                trgt_preds = tuple(target['output'])
                self.model[src_preds].append(trgt_preds)

        for src_preds in self.model:
            self.model[src_preds] = Counter(self.model[src_preds])
        return self.model


    def predict(self, source):
        src_triples = utils.split_triples(source)
        # source predicates
        src_preds = tuple(sorted([t[1] for t in src_triples]))

        if src_preds in self.model:
            predicates = max(self.model[src_preds].items(), key=operator.itemgetter(1))[0]

            # ordtriples = []
            # for predicate in list(predicates):
            #     for triple in src_triples:
            #         if triple[1] == predicate:
            #             ordtriples.append(triple)
            #             break
            #
            # ordtriples = utils.delexicalize(ordtriples)
            # target = []
            # for triple in ordtriples:
            #     target.append('<TRIPLE>')
            #     target.extend(triple)
            #     target.append('</TRIPLE>')
        else:
            predicates = [t[1] for t in src_triples]
        return predicates


    def evaluate(self, data):
        references, predictions = [], []
        num, dem = 0, 0
        for entry in data:
            if int(entry['size']) > 1:
                source = entry['source']
                prediction = ' '.join(self.predict(source))
                predictions.append(prediction)

                refs = []
                for target in entry['targets']:
                    # trg_triples = utils.split_triples(target['output'])
                    # trg_preds = [t[1] for t in trg_triples]
                    refs.append(' '.join(target['output']))
                references.append(refs)

                if prediction.strip() in refs:
                    num += 1
                dem += 1

        print('Accuracy: ', num / dem)
        return predictions, references


    def __call__(self, in_path, out_path):
        with open(in_path) as f:
            entries = [t.split() for t in f.read().split('\n')]
        result = [self.predict(triples) for triples in entries]

        with open(out_path, 'w') as f:
            doc = [' '.join(predicates) for predicates in result]
            f.write('\n'.join(doc))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        trainpath=sys.argv[3]
        model = MajorOrder(trainpath)
        model(in_path=in_path, out_path=out_path)
    else:
        path = '/roaming/tcastrof/emnlp2019/ordering/data'
        trainpath = os.path.join(path, 'train.json')
        model = MajorOrder(trainpath)

        print('Dev set:')
        devdata = json.load(open(os.path.join(path, 'dev.json')))
        predictions, references = model.evaluate(devdata)

        print('Test set:')
        testdata = json.load(open(os.path.join(path, 'test.json')))
        predictions, references = model.evaluate(testdata)