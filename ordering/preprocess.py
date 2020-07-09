__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/02/2019
Description:
    This script aims to extract the gold-standard ordered triple sets for the Discourse Ordering step.

    ARGS:
        [1] Path to the folder where WebNLG corpus is available (versions/v1.4/en)
        [2] Path to the folder where the data will be saved (Folder will be created in case it does not exist)

    EXAMPLE:
        python3 preprocess.py ../versions/v1.4/en ordering/
"""

import sys
sys.path.append('./')
sys.path.append('../')

import os
import load
import parsing
import utils

from superpreprocess import Preprocess

class Ordering(Preprocess):
    def __init__(self, data_path, write_path):
        super().__init__(data_path=data_path, write_path=write_path)

        self.traindata, self.vocab = self.load_simple(path=os.path.join(data_path, 'train.xml'))#, augment=False)
        self.devdata, _ = self.load_simple(path=os.path.join(data_path, 'dev.xml'))#, augment=False)
        self.testdata, _ = self.load_simple(path=os.path.join(data_path, 'test.xml'))#, augment=False)


    def load_simple(self, path):
        entryset = list(parsing.parse(path))

        data, size = [], 0
        invocab, outvocab = [], []

        for i, entry in enumerate(entryset):
            progress = round(float(i) / len(entryset), 2)
            print('Progress: {0}'.format(progress), end='   \r')
            try:
                # triples greater than 1
                if len(entry.modifiedtripleset) > 1:
                    # process source
                    tripleset = []
                    for i, triple in enumerate(entry.modifiedtripleset):
                        striple = triple.key + ' ' + triple.value
                        tripleset.append((i, striple))
                    # given a fixed order by sorting the set of triples automatically (predicate - subject - object)
                    tripleset = sorted(tripleset, key=lambda x: x[1])
                    triples = [entry.modifiedtripleset[t[0]] for t in tripleset]

                    entitymap = {b:a for a, b in entry.entitymap_to_dict().items()}
                    source, _, entities = load.source(triples, entitymap, {})
                    invocab.extend(source)

                    targets = []
                    for lex in entry.lexEntries:
                        # process ordered tripleset
                        _, text, _ = load.snt_source(lex.orderedtripleset, entitymap, entities)
                        text = [w for w in text if w not in ['<SNT>', '</SNT>']]
                        trg_preds = [t[0] for t in utils.split_triples(text)]

                        target = { 'lid': lex.lid, 'comment': lex.comment, 'output': trg_preds }
                        targets.append(target)
                        outvocab.extend(trg_preds)

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


    def load_index(self, path):
        entryset = parsing.run_parser(path)

        data, size = [], 0
        invocab, outvocab = [], []

        for i, entry in enumerate(entryset):
            progress = round(float(i) / len(entryset), 2)
            print('Progress: {0}'.format(progress), end='   \r')
            try:
                # triples greater than 1
                if len(entry.modifiedtripleset) > 1:
                    # process source
                    tripleset = []
                    for i, triple in enumerate(entry.modifiedtripleset):
                        striple = triple.predicate + ' ' + triple.subject + ' ' + triple.object
                        tripleset.append((i, striple))
                    # given a fixed order by sorting the set of triples automatically (predicate - subject - object)
                    tripleset = sorted(tripleset, key=lambda x: x[1])
                    triples = [entry.modifiedtripleset[t[0]] for t in tripleset]

                    entitymap = {b:a for a, b in entry.entitymap_to_dict().items()}
                    source, _, entities = load.source(triples, entitymap, {})
                    invocab.extend(source)

                    targets = []
                    for lex in entry.lexEntries:
                        # process ordered tripleset
                        trg_idx = []
                        orderedtripleset = [item for sublist in lex.orderedtripleset for item in sublist]
                        for sorted_triple in orderedtripleset:
                            for i, src_triple in enumerate(triples):
                                if sorted_triple.subject == src_triple.subject and \
                                                sorted_triple.predicate == src_triple.predicate and \
                                                sorted_triple.object == src_triple.object and str(i+1) not in trg_idx:
                                    trg_idx.append(str(i+1))

                        target = { 'lid': lex.lid, 'comment': lex.comment, 'output': trg_idx }
                        targets.append(target)
                        outvocab.extend(trg_idx)

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
    # write_path='/roaming/tcastrof/emnlp2019/ordering'
    # data_path = '../versions/v1.4/en'

    data_path = sys.argv[1]
    write_path = sys.argv[2]
    s = Ordering(data_path=data_path, write_path=write_path)
    s()

