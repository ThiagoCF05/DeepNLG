__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 26/03/2019
Description:
    This script aims to perform the Majority implementation of the Text Structuring step.

    ARGS:
        [1] Path to the textual file where the inputs are
        [2] Path to the textual file where the outputs should be saved
        [3] Path to train set

    EXAMPLE:
        python3 major.py structing/dev.eval structing/dev.major.out structing/data/train.json
"""

import sys
sys.path.append('../')
sys.path.append('./')
import copy
import json
import operator
import os
import utils

from collections import Counter

class MajorStructing():
    def __init__(self, train_path):
        traindata = json.load(open(train_path))
        self.train(traindata)


    def train(self, data):
        self.model = {}
        for entry in data:
            src_triples = utils.split_triples(entry['source'])
            src_preds = tuple([t[1] for t in src_triples])
            if src_preds not in self.model:
                self.model[src_preds] = []
            for target in entry['targets']:
                trgt_preds = ' '.join(target['output'])
                self.model[src_preds].append(trgt_preds)

        for source in self.model:
            self.model[source] = Counter(self.model[source])
        return self.model


    def predict(self, source):
        triples = utils.delexicalize(utils.split_triples(source))
        src_preds = [t[1] for t in triples]

        start, end = 0, len(src_preds)
        target = []
        while start < len(src_preds):
            preds = tuple(src_preds[start:end])
            if len(preds) == 1:
                struct = '<SNT> ' + preds[0] + ' </SNT>'
                target.append(struct)

                start = copy.copy(end)
                end = len(src_preds)
            elif preds in self.model:
                struct = max(self.model[preds].items(), key=operator.itemgetter(1))[0]
                target.append(struct)

                start = copy.copy(end)
                end = len(src_preds)
            else:
                end -= 1

        # if src_preds in self.model:
        #     target = max(self.model[src_preds].items(), key=operator.itemgetter(1))[0]
        #     target = target.split()
        # else:
        #     target = []
        #     for pred in src_preds:
        #         target.append('<SNT>')
        #         target.append(pred)
        #         target.append('</SNT>')

        return target


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
                    trgt_preds = ' '.join(target['output'])
                    refs.append(trgt_preds)
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
        trainpath = sys.argv[3]
        model = MajorStructing(trainpath)
        model(in_path=in_path, out_path=out_path)
    else:
        path = '/roaming/tcastrof/emnlp2019/structing/data'
        trainpath = os.path.join(path, 'train.json')
        model = MajorStructing(trainpath)
        print('Dev set:')
        devdata = json.load(open(os.path.join(path, 'dev.json')))
        predictions, references = model.evaluate(devdata)

        print('Test set:')
        testdata = json.load(open(os.path.join(path, 'test.json')))
        predictions, references = model.evaluate(testdata)