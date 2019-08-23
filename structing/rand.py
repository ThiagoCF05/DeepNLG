__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 26/03/2019
Description:
    This script aims to perform the Random implementation of the Text Structuring step.

    ARGS:
        [1] Path to the textual file where the inputs are
        [2] Path to the textual file where the outputs should be saved

    EXAMPLE:
        python3 rand.py structing/dev.eval structing/dev.rand.out
"""

import sys
sys.path.append('./')
sys.path.append('../')
import json
import os
import utils

from random import randint

class RandomStruct():
    def predict(self, source):
        triples = utils.split_triples(source)
        predicates = [t[1] for t in triples]

        start, end = 0, -1
        intervals = []
        while end < len(triples) and len(triples) > 0:
            end = randint(start+1, len(triples))
            intervals.append((start, end))
            start = end

        struct = []
        for interval in intervals:
            start, end = interval
            struct.append('<SNT>')
            for predicate in predicates[start:end]:
                struct.append(predicate)
            struct.append('</SNT>')
        return struct


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
    path = '/roaming/tcastrof/emnlp2019/structing/data'
    model = RandomStruct()

    if len(sys.argv) > 1:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        model(in_path=in_path, out_path=out_path)
    else:
        print('Dev set:')
        devdata = json.load(open(os.path.join(path, 'dev.json')))
        model.evaluate(devdata)

        print('Test set:')
        testdata = json.load(open(os.path.join(path, 'test.json')))
        model.evaluate(testdata)