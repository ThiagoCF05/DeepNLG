__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 26/03/2019
Description:
    This script aims to extract the gold-standard templates for the lexicalization step.

    ARGS:
        [1] Path to the folder where WebNLG corpus is available (versions/v1.4/en)
        [2] Path to the folder where the data will be saved (Folder will be created in case it does not exist)
        [3] Path to the stanford parser (which can be downloaded here: https://stanfordnlp.github.io/CoreNLP/)

    EXAMPLE:
        python3 preprocess.py ../versions/v1.4/en lexicalization stanford_path
"""

import sys
sys.path.append('./')
sys.path.append('../')
from superpreprocess import Preprocess

import copy
import json
import load
import parsing
import os


class Lexicalization(Preprocess):
    def __init__(self, data_path, write_path):
        super().__init__(data_path=data_path, write_path=write_path)

        self.traindata, self.vocab = self.load(os.path.join(data_path, 'train.xml'))
        self.devdata, _ = self.load(os.path.join(data_path, 'dev.xml'))
        self.testdata, _ = self.load(os.path.join(data_path, 'test.xml'))


    def __call__(self):
        self.run(traindata=self.traindata, devdata=self.devdata, testdata=self.testdata)


    def load(self, path):
        entryset = list(parsing.parse(path))

        data, size = [], 0
        invocab, outvocab = [], []
        nerrors = 0
        for i, entry in enumerate(entryset):
            progress = round(i / len(entryset), 2)
            print('Progress: {0} \t Errors: {1}'.format(progress, round(nerrors / len(entryset), 2)), end='\r')
            entitymap = {b:a for a, b in entry.entitymap_to_dict().items()}

            visited = []
            for lex in entry.lexEntries:
                # process ordered tripleset
                source, delex_source, _ = load.snt_source(lex.orderedtripleset, entitymap, {})

                if source not in visited:
                    visited.append(source)
                    invocab.extend(source)

                    targets = []
                    for lex2 in entry.lexEntries:
                        _, target, entities = load.snt_source(lex2.orderedtripleset, entitymap, {})

                        if delex_source == target:
                            try:
                                template = lex2.lexicalization.split()
                                for i, word in enumerate(template):
                                    if word in entities:
                                        template[i] = entities[word]
                                target = { 'lid': lex.lid, 'comment': lex.comment, 'output': template }
                                targets.append(target)
                                outvocab.extend(template)
                            except:
                                nerrors += 1
                                print('Parsing Error...')

                    data.append({
                        'eid': entry.eid,
                        'category': entry.category,
                        'size': entry.size,
                        'source': source,
                        'targets': targets })
                    size += len(targets)

        invocab.append('<unk>')
        outvocab.append('<unk>')

        invocab = list(set(invocab))
        outvocab = list(set(outvocab))
        vocab = { 'input': invocab, 'output': outvocab }

        print('Path:', path, 'Size: ', size)
        return data, vocab


if __name__ == '__main__':
    # data_path = '../versions/v1.4/en'
    # write_path='/roaming/tcastrof/emnlp2019/templatization'

    data_path = sys.argv[1]
    write_path = sys.argv[2]
    temp = Lexicalization(data_path=data_path, write_path=write_path)
    temp()

