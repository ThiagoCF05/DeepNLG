__author__='thiagocastroferreira'

import json

from random import shuffle

if __name__ == '__main__':
    entries = json.load(open('questionaire/gold.json'))
    entries = [e for e in entries if e['humaneval']]

    sizes = set([e['size'] for e in entries])
    categories = set([e['category'] for e in entries])

    ids = []
    for category in categories:
        for size in sizes:
            f = [e for e in entries if e['size'] == size and e['category'] == category]
            shuffle(f)
            ids.append(f[0]['eid'])

    ids = sorted(ids, key=lambda x: int(x.replace('Id', '')))
    for id in ids:
        print(id)