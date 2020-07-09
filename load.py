__author__='thiagocastroferreira'


def source(tripleset, entitymap, entities={}):
    src, delexsrc = [], []
    for keyvalue in tripleset:
        key = '_'.join(keyvalue.key.split())
        value = '_'.join(keyvalue.value.split())

        src.append('<TRIPLE>')
        src.append(key)
        src.append(value)
        src.append('</TRIPLE>')

        delexsrc.append('<TRIPLE>')
        # KEY
        delexsrc.append(key)

        # VALUE
        if keyvalue.value in entitymap:
            if entitymap[keyvalue.value] not in entities:
                entity = 'ENTITY-' + str(len(list(entities.keys())) + 1)
                entities[entitymap[keyvalue.value]] = entity
            else:
                entity = entities[entitymap[keyvalue.value]]
            delexsrc.append(entity)
        else:
            delexsrc.append(keyvalue.value)

        delexsrc.append('</TRIPLE>')
    # src.append('eos')
    # delexsrc.append('eos')
    return src, delexsrc, entities


def snt_source(tripleset, entitymap, entities):
    aggregation = []
    delex_aggregation = []

    for sentence in tripleset:
        snt, delex_snt = ['<SNT>'], ['<SNT>']
        for skeyvalue in sentence:
            key = '_'.join(skeyvalue.key.split())
            value = '_'.join(skeyvalue.value.split())

            snt.append('<TRIPLE>')
            snt.append(key)
            snt.append(value)
            snt.append('</TRIPLE>')

            delex_snt.append('<TRIPLE>')

            # KEY
            delex_snt.append(key)

            # VALUE
            if skeyvalue.value in entitymap:
                if entitymap[skeyvalue.value] not in entities:
                    entity = 'ENTITY-' + str(len(list(entities.keys())) + 1)
                    entities[entitymap[skeyvalue.value]] = entity
                else:
                    entity = entities[entitymap[skeyvalue.value]]
                delex_snt.append(entity)
            else:
                delex_snt.append(skeyvalue.value)

            delex_snt.append('</TRIPLE>')

        snt.append('</SNT>')
        aggregation.extend(snt)

        delex_snt.append('</SNT>')
        delex_aggregation.extend(delex_snt)

    # aggregation = ['bos'] + aggregation + ['eos']
    # delex_aggregation = ['bos'] + delex_aggregation + ['eos']
    return aggregation, delex_aggregation, entities