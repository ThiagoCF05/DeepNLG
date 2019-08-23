__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 26/03/2019
Description:
    This script is responsible to retrieve the triples based on the output of the Discourse Ordering
    and Text Structuring steps of the pipeline approach. Moreover, it also wikifies a Lexicalization template

    ARGS:
        [1] Path to the textual file where the inputs of Discourse Ordering/Text Structuring are
        [2] Path to the textual file where the outputs should be saved
        [3] Flag to indicate the pipeline step: Discourse Ordering -> ordering / Text Structuring -> structing / Lexicalization -> lexicalization
        [4] File name to save the output

    EXAMPLE:
        python3 mapping.py ordering/dev.eval ordering/dev.ordering.postprocessed ordering ordering/dev.ordering.mapped
"""

import sys
import utils


def orderout2structin(ordering_out, triples):
    ord_triples = []
    if len(triples) == 1:
        ord_triples.extend(triples)
    else:
        added = []
        for predicate in ordering_out:
            for i, triple in enumerate(triples):
                if predicate.strip() == triple[1].strip() and i not in added:
                    ord_triples.append(triple)
                    added.append(i)
                    break

    return ' '.join(utils.join_triples(ord_triples))


def orderout2structin_simple(ordering_out, triples):
    ord_triples = []
    if len(triples) == 1:
        ord_triples.extend(triples)
    else:
        for idx in ordering_out:
            try:
                ord_triples.append(triples[int(idx)-1])
            except:
                pass

    return ' '.join(utils.join_triples(ord_triples))


def structout2lexin(struct_out, triples):
    sentences, snt = [], []
    for w in struct_out:
        if w.strip() not in ['<SNT>', '</SNT>']:
            snt.append(w.strip())

        if w.strip() == '</SNT>':
            sentences.append(snt)
            snt = []

    struct, struct_unit = [], []
    if len(triples) == 1:
        struct.append(triples)
    else:
        added = []
        for snt in sentences:
            for predicate in snt:
                for i, triple in enumerate(triples):
                    if predicate.strip() == triple[1].strip() and i not in added:
                        struct_unit.append(triple)
                        added.append(i)
                        break
            struct.append(struct_unit)
            struct_unit = []
    return ' '.join(utils.join_struct(struct))


def structout2lexin_simple(struct_out, triples):
    struct, snt = [], []
    pos = 0
    for w in struct_out:
        if w.strip() == '<TRIPLE>':
            try:
                snt.append(triples[pos])
                pos += 1
            except:
                pass
        elif w.strip() == '</SNT>':
            struct.append(snt)
            snt = []
    return ' '.join(utils.join_struct(struct))


def lexout2regin(lex_out, triples):
    entities = utils.entity_mapping(triples)
    for i, w in enumerate(lex_out):
        if w.strip() in entities:
            lex_out[i] = entities[w]
    return ' '.join(lex_out)


def run(out_path, entries_path, task):
    with open(out_path) as f:
        outputs = f.read().split('\n')
    outputs = [out.split() for out in outputs]

    with open(entries_path) as f:
        entries = f.read().split('\n')

    entries = [utils.split_triples(t.split()) for t in entries]
    for i, entry in enumerate(entries):
        if task == 'ordering':
            yield orderout2structin(ordering_out=outputs[i], triples=entry)
        elif task == 'structing':
            yield structout2lexin(struct_out=outputs[i], triples=entry)
        else:
            yield lexout2regin(lex_out=outputs[i], triples=entry)

if __name__ == '__main__':
    entries_path = sys.argv[1]
    out_path = sys.argv[2]
    task = sys.argv[3]

    write_path = sys.argv[4]
    result = run(out_path=out_path, entries_path=entries_path, task=task)

    with open(write_path, 'w') as f:
        f.write('\n'.join(result))