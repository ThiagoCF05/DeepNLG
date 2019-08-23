__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 26/03/2019
Description:
    This script aims to perform the Random implementation of the Discourse Ordering step.

    ARGS:
        [1] Path to the textual file where the inputs are
        [2] Path to the textual file where the outputs should be saved

    EXAMPLE:
        python3 rand.py ordering/dev.eval ordering/dev.rand.out
"""

import sys
sys.path.append('../')
sys.path.append('./')
import utils
import json
import os

from itertools import permutations
from random import randint

class RandomOrder():
    def predict(self, source):
        src_triples = utils.delexicalize(utils.split_triples(source))
        src_preds = [t[1] for t in src_triples]
        perm = list(permutations(src_preds))
        pos = randint(0, len(perm)-1)

        predicates = perm[pos]
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
    model = RandomOrder()

    if len(sys.argv) > 1:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        model(in_path=in_path, out_path=out_path)
    else:
        print('Dev set:')
        path = '/roaming/tcastrof/emnlp2019/ordering/data'
        devdata = json.load(open(os.path.join(path, 'dev.json')))
        model.evaluate(devdata)

        print('Test set:')
        path = '/roaming/tcastrof/emnlp2019/ordering/data'
        testdata = json.load(open(os.path.join(path, 'test.json')))
        model.evaluate(testdata)