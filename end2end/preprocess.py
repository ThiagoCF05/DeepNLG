__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 26/03/2019
Description:
    Tripleset to text

    ARGS:
        [1] Path to the folder where WebNLG corpus is available (versions/v1.4/en)
        [2] Path to the folder where the data will be saved (Folder will be created in case it does not exist)

    EXAMPLE:
        python3 preprocess.py ../versions/v1.4/en end2end
"""

import sys
sys.path.append('./')
sys.path.append('../')

import load
import os
import parsing
import nltk

from superpreprocess import Preprocess
from itertools import permutations
from random import randint

STANFORD_PATH=r'~/stanford/stanford-corenlp-full-2018-02-27'

class End2End(Preprocess):
    def __init__(self, data_path, write_path):
        super().__init__(data_path=data_path, write_path=write_path)

        self.traindata, self.vocab = self.load_simple(path=os.path.join(data_path, 'train.xml'))#, augment=True)
        self.devdata, _ = self.load_simple(path=os.path.join(data_path, 'dev.xml'))#, augment=False)
        self.testdata, _ = self.load_simple(path=os.path.join(data_path, 'test.xml'))#, augment=False)


    def tokenize(self, text):
        text = text.replace('@', ' ')
        tokens = []
        # tokenizing text
        try:
            for snt in nltk.sent_tokenize(text, language='dutch'):
                sentence = nltk.word_tokenize(snt, language='dutch')
                tokens.extend(sentence)
        except:
            print('Parsing error...')

        return tokens


    def load(self, path, augment=True):
        entryset = parsing.parse(path)

        data, size = [], 0
        invocab, outvocab = [], []

        for i, entry in enumerate(entryset):
            progress = round(float(i) / len(entryset), 2)
            print('Progress: {0}'.format(progress), end='   \r')
            try:
                # process source
                entitymap = {b:a for a, b in entry.entitymap_to_dict().items()}
                source, _, entities = load.source(entry.modifiedtripleset, entitymap, {})
                invocab.extend(source)

                targets = []
                for lex in entry.lexEntries:
                    # process ordered tripleset
                    text = self.tokenize(text=lex.text)

                    target = { 'lid': lex.lid, 'comment': lex.comment, 'output': text, 'text': lex.text.replace('@', ' ') }
                    targets.append(target)
                    outvocab.extend(text)

                data.append({
                    'eid': entry.eid,
                    'category': entry.category,
                    'augmented': False,
                    'size': entry.size,
                    'source': source,
                    'targets': targets })
                size += len(targets)

                # choose the original order and N permutations such as N = len(tripleset)-1
                if augment:
                    triplesize = len(entry.modifiedtripleset)
                    perm = list(permutations(entry.modifiedtripleset))
                    perm = [load.source(src, entitymap, {}) for src in perm]
                    entitylist = [w[2] for w in perm]
                    perm = [w[0] for w in perm]

                    taken = []
                    # to augment the corpus, pick the minumum between the number of permutations - 1 or 49
                    X = min(len(perm)-1, 49)
                    for _ in range(X):
                        found = False
                        while not found and triplesize != 1:
                            pos = randint(0, len(perm)-1)
                            src, entities = perm[pos], entitylist[pos]

                            if pos not in taken and src != source:
                                taken.append(pos)
                                found = True

                                targets = []
                                for lex in entry.lexEntries:
                                    # process ordered tripleset
                                    text = self.tokenize(text=lex.text)

                                    target = { 'lid': lex.lid, 'comment': lex.comment, 'output': text, 'text': lex.text.replace('@', ' ') }
                                    targets.append(target)
                                    outvocab.extend(text)

                                data.append({
                                    'eid': entry.eid,
                                    'category': entry.category,
                                    'augmented': True,
                                    'size': entry.size,
                                    'source': src,
                                    'targets': targets })
                                size += len(targets)
            except:
                print('Preprocessing error...')

        invocab.append('unk')
        outvocab.append('unk')

        invocab = list(set(invocab))
        outvocab = list(set(outvocab))
        vocab = { 'input': invocab, 'output': outvocab }

        print('Path:', path, 'Size: ', size)
        return data, vocab


    def load_simple(self, path):
        entryset = list(parsing.parse(path))

        data, size = [], 0
        invocab, outvocab = [], []

        for i, entry in enumerate(entryset):
            progress = round(float(i) / len(entryset), 2)
            print('Progress: {0}'.format(progress), end='   \r')
            try:
                # process source
                tripleset = []
                for i, triple in enumerate(entry.modifiedtripleset):
                    striple = triple.key + ' ' + triple.value
                    tripleset.append((i, striple))
                # given a fixed order by sorting the set of triples automatically (predicate - subject - object)
                tripleset = sorted(tripleset, key=lambda x: x[0])
                triples = [entry.modifiedtripleset[t[0]] for t in tripleset]

                entitymap = {b:a for a, b in entry.entitymap_to_dict().items()}
                source, _, entities = load.source(triples, entitymap, {})
                invocab.extend(source)

                targets = []
                for lex in entry.lexEntries:
                    # process ordered tripleset
                    text = self.tokenize(text=lex.text)

                    target = { 'lid': lex.lid, 'comment': lex.comment, 'output': text, 'text': lex.text.replace('@', ' ') }
                    targets.append(target)
                    outvocab.extend(text)

                data.append({
                    'eid': entry.eid,
                    'category': entry.category,
                    'augmented': False,
                    'size': entry.size,
                    'source': source,
                    'targets': targets })
                size += len(targets)
            except:
                print('Preprocessing error...')

        invocab.append('unk')
        outvocab.append('unk')

        invocab = list(set(invocab))
        outvocab = list(set(outvocab))
        vocab = { 'input': invocab, 'output': outvocab }

        print('Path:', path, 'Size: ', size)
        return data, vocab


    def __call__(self):
        self.run(traindata=self.traindata, devdata=self.devdata, testdata=self.testdata)


if __name__ == '__main__':
    # write_path='/roaming/tcastrof/emnlp2019/end2end'
    # data_path = '../versions/v1.4/en'

    data_path = sys.argv[1]
    write_path = sys.argv[2]
    s = End2End(data_path=data_path, write_path=write_path)
    s()

