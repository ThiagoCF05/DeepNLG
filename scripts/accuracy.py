__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 27/03/2019
Description:
    Evaluate the accuracy of a module in predicting at least one of the gold-standards

    ARGS:
        [0] Reference path
        [1] Predictions path

    EXAMPLE:
        python3 accuracy.py reference_path prediction_path
"""

import sys
import os

def load_references(ref_path):
    references = []
    for i in range(1, 6):
        if os.path.exists(ref_path + str(i)):
            with open(ref_path + str(i)) as f:
                refs = f.read().split('\n')

            for j, ref in enumerate(refs):
                if j == len(references):
                    references.append([ref])
                else:
                    references[j].append(ref)

    for i, refs in enumerate(references):
        references[i] = [ref.strip() for ref in refs if ref.strip() != '']
    return references


def load_predictions(pred_path):
    with open(pred_path) as f:
        predictions = f.read().split('\n')

    return [p.strip() for p in predictions]


def evaluate(predictions, references):
    num, dem = 0, 0
    for i, prediction in enumerate(predictions):
        if prediction in references[i]:
            num += 1
        dem += 1
    return num / dem


if __name__ == '__main__':
    ref_path = sys.argv[1]
    pred_path = sys.argv[2]

    references = load_references(ref_path)
    predictions = load_predictions(pred_path)

    result = evaluate(predictions, references)
    print(result)