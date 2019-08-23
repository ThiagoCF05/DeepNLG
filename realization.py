__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 26/03/2019
Description:
    This script aims to perform the Textual Realization step of our pipeline approach

    ARGS:
        [1] Path to the textual file where the inputs are
        [2] Path to the textual file where the outputs should be saved
        [3] Path to the vocabulary (in JSON)

    EXAMPLE:
        python3 realization.py pipeline/dev.reg pipeline/dev.realized lexicalization/surfacevocab.json
"""

import sys
import json
import operator

SURFACE_PATH='/roaming/tcastrof/emnlp2019/lexicalization/surfacevocab.json'

class Realization():
    def __init__(self, rule_path):
        self.surface_rules = json.load(open(rule_path))


    def realize(self, entry):
        template = entry
        stemplate = entry.split()

        for i, token in enumerate(stemplate):
            if 'VP[' == token[:3]:
                try:
                    rule = token.strip() + ' ' + stemplate[i+1].strip()
                    try:
                        surface_rule = max(self.surface_rules[rule].items(), key=operator.itemgetter(1))[0]
                        template = template.replace(rule, surface_rule)
                    except:
                        template = template.replace(rule, stemplate[i+1].strip())
                except:
                    template = template.replace(token.strip(), ' ')
            elif 'DT[' == token[:3]:
                rule = token.strip() + ' ' + stemplate[i+1].strip()
                if token.strip() == 'DT[form=undefined]':
                    if stemplate[i+2].strip().lower()[0] in ['a', 'e', 'i', 'o', 'u']:
                        template = template.replace(rule, 'an')
                    else:
                        template = template.replace(rule, 'a')
                else:
                    try:
                        surface_rule = max(self.surface_rules[rule].items(), key=operator.itemgetter(1))[0]
                        template = template.replace(rule, surface_rule)
                    except:
                        template = template.replace(rule, stemplate[i+1].strip())
        template = template.replace('-LRB-', '(').replace('-RRB-', ')')
        return template


    def __call__(self, in_path, out_path):
        with open(in_path) as f:
            entries = f.read().split('\n')

        result = [self.realize(entry) for entry in entries]
        with open(out_path, 'w') as f:
            f.write('\n'.join(result))


if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    SURFACE_PATH=sys.argv[3]

    model = Realization(SURFACE_PATH)
    model(in_path=in_path, out_path=out_path)