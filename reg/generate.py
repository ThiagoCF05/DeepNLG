__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/02/2019
Description:
    This script aims to generate the referring expressions.

    ARGS:
        [1] Path to the file with the Lexicalization step output
        [2] Path to the file with the Discourse Ordering step output
        [3] Path to the file where the output will be saved
        [4] Flag to specify the model: NeuralREG -> neuralreg / OnlyName -> onlynames
        [5] Path to the trained model

    EXAMPLE:
        python3 generate.py dev.lex.postprocessed dev.ordering.mapped dev.reg neuralreg reg/model1.dy
"""

import sys
sys.path.append('./')
sys.path.append('../')

from neuralreg import NeuralREG
import utils
import re

class REG():
    def __init__(self, model, model_path):
        self.model = model.strip()
        if model == 'neuralreg':
            config = {
                'LSTM_NUM_OF_LAYERS':1,
                'EMBEDDINGS_SIZE':300,
                'STATE_SIZE':512,
                'ATTENTION_SIZE':512,
                'DROPOUT':0.2,
                'GENERATION':30,
                'BEAM_SIZE':5,
                'BATCH_SIZE': 80,
                'EPOCHS': 60,
                'EARLY_STOP': 20
            }

            path = '/roaming/tcastrof/emnlp2019/reg'
            self.neuralreg = NeuralREG(path=path, config=config)
            self.neuralreg.populate(model_path)


    def realize_date(self, entity):
        regex='([0-9]{4})-([0-9]{2})-([0-9]{2})'
        dates = re.findall(regex,entity)
        if len(dates) > 0:
            year, month, day = dates[0]

            month = int(month)
            if month == 1:
                month = 'January'
            elif month == 2:
                month = 'February'
            elif month == 3:
                month = 'March'
            elif month == 4:
                month = 'April'
            elif month == 5:
                month = 'May'
            elif month == 6:
                month = 'June'
            elif month == 7:
                month = 'July'
            elif month == 8:
                month = 'August'
            elif month == 9:
                month = 'September'
            elif month == 10:
                month = 'October'
            elif month == 11:
                month = 'November'
            elif month == 12:
                month = 'December'

            refex = '{0} {1}, {2}'.format(month, str(int(day)), str(int(year)))
            return True, refex
        return False, ''


    def realize(self, entry, entity_map):
        entry = entry.split()
        pre_context = ['eos']
        for i, token in enumerate(entry):
            if token.strip() in entity_map:
                entity = entity_map[token.strip()]
                isDate, refex = self.realize_date(entity)
                if not isDate:
                    try:
                        isTrain = '_'.join(entity.split()) in self.neuralreg.vocab['input']
                    except:
                        isTrain = False
                    if entity[0] in ['\'', '\"'] or self.model != 'neuralreg' or not isTrain:
                        refex = entity.replace('_', ' ').replace('\"', ' ').replace('\'', ' ')
                    else:
                        try:
                            refex = str(int(entity))
                        except ValueError:
                            pos_context = []
                            for j in range(i+1, len(entry)):
                                if entry[j].strip() in entity_map:
                                    pos_context.append(entity_map[entry[j].strip()])
                                else:
                                    pos_context.append(entry[j].strip().lower())
                            pos_context.append('eos')
                            candidates = self.neuralreg(pre_context=pre_context, pos_context=pos_context, entity=entity, beam=self.neuralreg.config.beam)
                            refex = ' '.join(candidates[0]).replace('eos', '').strip()

                entry[i] = refex
                pre_context.append(entity)
            else:
                pre_context.append(token.lower())
        return entry


    def __call__(self, in_path, order_path, out_path):
        with open(in_path) as f:
            entries = f.read().split('\n')

        with open(order_path) as f:
            ordered_triples = [utils.split_triples(t.split()) for t in f.read().split('\n')]

        entity_maps = [utils.entity_mapping(t) for t in ordered_triples]
        result = []
        for i, entry in enumerate(entries):
            print('Progress: ', round(i / len(entries), 2), end='\r')
            result.append(self.realize(entry, entity_maps[i]))
        # result = [self.realize(entry, entity_maps[i]) for i, entry in enumerate(entries)]
        with open(out_path, 'w') as f:
            out = [' '.join(predicates) for predicates in result]
            f.write('\n'.join(out))


if __name__ == '__main__':
    path = '/roaming/tcastrof/emnlp2019/lexicalization/surfacevocab.json'

    in_path = sys.argv[1]
    order_path = sys.argv[2]
    out_path = sys.argv[3]
    model = sys.argv[4]
    model_path = sys.argv[5]
    model = REG(model=model, model_path=model_path)
    model(in_path=in_path, order_path=order_path, out_path=out_path)