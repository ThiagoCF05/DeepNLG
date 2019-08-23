__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 26/03/2019
Description:
    This script aims to perform the Majority implementation of the Lexicalization step.

    ARGS:
        [1] Path to the textual file where the inputs are
        [2] Path to the textual file where the outputs should be saved
        [3] Path to train set

    EXAMPLE:
        python3 major.py lexicalization/dev.eval lexicalization/dev.major.out lexicalization/data/train.json
"""

import sys
sys.path.append('./')
sys.path.append('../')
import copy
import json
import os
import utils
import operator

from collections import Counter

class MajorLexicalization():
    def __init__(self, trainpath):
        traindata = json.load(open(trainpath))
        self.train(traindata)


    def train(self, data):
        self.model = {}
        for entry in data:
            triples = utils.delexicalize_struct(utils.split_struct(entry['source']))
            source= []
            for snt in triples:
                sentence = ' '.join(['<SNT>'] + [t[1] for t in snt] + ['</SNT>'])
                source.append(sentence)

            source = tuple(source)
            if source not in self.model:
                self.model[source] = []

            for target in entry['targets']:
                output = ' '.join(target['output'])
                self.model[source].append(output)

        for snt in self.model:
            self.model[snt] = Counter(self.model[snt])
        return self.model


    def track_entity(self, sentences):
        entities, entitytag, entity_pos = {}, {}, 1
        for snt in sentences:
            for triple in snt:
                agent = triple[0]
                if agent not in entitytag:
                    entitytag[agent] = 'ENTITY-' + str(entity_pos)
                    entities['ENTITY-' + str(entity_pos)] = agent
                    entity_pos += 1

                patient = triple[-1]
                if patient not in entitytag:
                    entitytag[patient] = 'ENTITY-' + str(entity_pos)
                    entities['ENTITY-' + str(entity_pos)] = patient
                    entity_pos += 1
        return entities, entitytag


    def predict(self, source):
        sentences = utils.split_struct(source)
        triples = utils.delexicalize_struct(sentences)
        struct= []
        for snt in triples:
            sentence = ' '.join(['<SNT>'] + [t[1] for t in snt] + ['</SNT>'])
            struct.append(sentence)

        target = []
        # Try to extract a full template
        start, end, templates = 0, len(struct), []
        while start < len(struct):
            snts = tuple(struct[start:end])
            entities, _ = self.track_entity(sentences[start:end])

            if snts in self.model:
                template = max(self.model[snts].items(), key=operator.itemgetter(1))[0].split()
                for i, w in enumerate(template):
                    if w in entities:
                        template[i] = entities[w]
                target.extend(template)

                start = copy.copy(end)
                end = len(struct)
            else:
                end -= 1

                # jump a triple if it is not on training set
                if start == end:
                    start += 1
                    end = len(struct)

        _, entitytag = self.track_entity(sentences)
        for i, w in enumerate(target):
            if w in entitytag:
                target[i] = entitytag[w]
        return target


    def evaluate(self, data):
        references, predictions = [], []
        num, dem = 0, 0
        for entry in data:
            source = entry['source']
            prediction = ' '.join(self.predict(source))
            predictions.append(prediction)
            # predictions.append(utils.delexicalize_verb(prediction))

            refs = []
            for target in entry['targets']:
                output = ' '.join(target['output'])
                refs.append(output)
                # refs.append(utils.delexicalize_verb(output))
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
            out = [' '.join(predicates) for predicates in result]
            f.write('\n'.join(out))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        trainpath = sys.argv[3]
        model = MajorLexicalization(trainpath)
        model(in_path=in_path, out_path=out_path)
    else:
        path = '/roaming/tcastrof/emnlp2019/lexicalization/data'
        trainpath = os.path.join(path, 'train.json')
        model = MajorLexicalization(trainpath)
        print('Dev set:')
        devdata = json.load(open(os.path.join(path, 'dev.json')))
        predictions, references = model.evaluate(devdata)

        with open('predictions', 'w') as f:
            f.write('\n'.join(predictions))

        nfiles = max([len(refs) for refs in references])
        for i in range(5):
            with open('reference' + str(i+1), 'w') as f:
                for refs in references:
                    if i < len(refs):
                        f.write(refs[i])
                    f.write('\n')

        nematus = '/roaming/tcastrof/workspace/nematus/data/multi-bleu-detok.perl'
        command = 'perl ' + nematus + ' reference1 reference2 reference3 reference4 reference5 < predictions'
        os.system(command)

        os.remove('reference1')
        os.remove('reference2')
        os.remove('reference3')
        os.remove('reference4')
        os.remove('reference5')
        os.remove('predictions')

        print('Test set:')
        testdata = json.load(open(os.path.join(path, 'test.json')))
        predictions, references = model.evaluate(testdata)

        with open('predictions', 'w') as f:
            f.write('\n'.join(predictions))

        nfiles = max([len(refs) for refs in references])
        for i in range(5):
            with open('reference' + str(i+1), 'w') as f:
                for refs in references:
                    if i < len(refs):
                        f.write(refs[i])
                    f.write('\n')

        nematus = '/roaming/tcastrof/workspace/nematus/data/multi-bleu-detok.perl'
        command = 'perl ' + nematus + ' reference1 reference2 reference3 reference4 reference5 < predictions'
        os.system(command)

        os.remove('reference1')
        os.remove('reference2')
        os.remove('reference3')
        os.remove('reference4')
        os.remove('reference5')
        os.remove('predictions')