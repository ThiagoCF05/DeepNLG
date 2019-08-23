__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 03/05/2019
Description:
    Creating a json file with all results for comparison
"""

import sys
sys.path.append('../')
import os
import json
import re

import xml.etree.ElementTree as ET
from xml.dom import minidom
import utils

path='/roaming/tcastrof/emnlp2019'
gold_path=os.path.join(path, 'end2end/data/test.json')

class HumanEval:
    def __init__(self):
        self.gold = json.load(open(gold_path))

        self.load_final_results()
        self.gold = self.save_results()


    def load_final_results(self):
        with open('results/sota/ADAPTcentre-submission.txt') as f:
            self.adapt = f.read().split('\n')

        with open('results/sota/melbourne_submission.txt') as f:
            self.melbourne = f.read().split('\n')

        with open('results/sota/upf-forge_submission.txt') as f:
            self.upfforge = f.read().split('\n')

        with open('results/pipeline/rand/test.out') as f:
            self.rand = f.read().lower().split('\n')

        with open('results/pipeline/major/test.out') as f:
            self.major = f.read().lower().split('\n')

        with open('results/pipeline/transformer/test.out') as f:
            self.transformer = f.read().lower().split('\n')

        with open('results/pipeline/rnn/test.out') as f:
            self.rnn = f.read().lower().split('\n')

        with open('results/end2end/transformer1/test.out.post') as f:
            self.e2etransformer = f.read().lower().split('\n')

        with open('results/end2end/rnn/test.out.post') as f:
            self.e2ernn = f.read().lower().split('\n')

        with open('human/sample-ids.txt') as f:
            self.human_ids = [int(w) for w in f.read().split('\n')]


    def save_results(self):
        for i, entry in enumerate(self.gold):
            eid = int(entry['eid'].replace('Id', ''))

            # state-of-the-art
            entry['results'] = {}
            entry['results']['upfforge'] = self.upfforge[eid-1]
            entry['results']['melbourne'] = self.melbourne[eid-1]
            entry['results']['adapt'] = self.adapt[eid-1]
            # pipeline
            entry['results']['rand'] =  self.rand[i]
            entry['results']['major'] = self.major[i]
            entry['results']['transformer'] = self.transformer[i]
            entry['results']['rnn'] = self.rnn[i]
            # end-to-end
            entry['results']['e2etransformer'] = self.e2etransformer[i]
            entry['results']['e2ernn'] = self.e2ernn[i]

            entry['humaneval'] = False
            if eid in self.human_ids:
                entry['humaneval'] = True
        json.dump(self.gold, open('questionaire/gold.json', 'w'))
        return self.gold


    def ordering_analysis(self, ordering):
        for i, entry in enumerate(self.gold):
            triples = utils.split_triples(entry['source'])

            num, visited = 0, []
            for triple in triples:
                for j, predicate in enumerate(ordering[i]):
                    if predicate == triple[1] and j not in visited:
                        num += 1
                        visited.append(j)
            # How many predicates in the modified tripleset are present in the result?
            entry['ordering'] = num
        return self.gold


    def structing_analysis(self, structing):
        for i, entry in enumerate(self.gold):
            triples = utils.split_triples(entry['source'])

            num, visited = 0, []
            for triple in triples:
                for j, predicate in enumerate(structing[i]):
                    if predicate == triple[1] and j not in visited:
                        num += 1
                        visited.append(j)
            # How many predicates in the modified tripleset are present in the result?
            entry['structing'] = num
        return self.gold


    def qualitative_analysis(self, tag, structs, write_file):
        tree = ET.parse('human/testdata_no_lex.xml')
        root = tree.getroot()

        entries = root.find('entries')
        for i, entry in enumerate(self.gold):
            eid = entry['eid']
            entry_xml = [entry_xml for entry_xml in entries.findall('entry') if entry_xml.attrib['eid'] == eid][0]

            if int(eid.replace('Id', '')) not in self.human_ids:
                entries.remove(entry_xml)
            else:
                del entry_xml.attrib['category']
                # del entry_xml.attrib['size']
                # remove double entries
                originaltriplesets = entry_xml.findall('originaltripleset')
                for originaltripleset in originaltriplesets:
                    entry_xml.remove(originaltripleset)

                # remove double entries
                modifiedtriplesets = entry_xml.findall('modifiedtripleset')
                for modifiedtripleset in modifiedtriplesets[1:]:
                    entry_xml.remove(modifiedtripleset)

                struct_xml = ET.SubElement(entry_xml, 'structure')
                struct_xml.text = structs[i].strip().replace('<SNT>', 'SNT').replace('</SNT>', 'SNT')

                text_xml = ET.SubElement(entry_xml, 'text')
                if tag == 'original':
                    text_xml.text = ' '.join(entry['targets'][0]['output']).lower()
                else:
                    text_xml.text = entry['results'][tag]

                questions = ET.SubElement(entry_xml, 'questions')
                question1 = ET.SubElement(questions, 'question1')
                question1.text = 'Is the predicted structure followed by the text?'
                question1.attrib['answer'] = 'y/n'

                question2 = ET.SubElement(questions, 'question2')
                question2.text = 'How many predicates in the modified tripleset are verbalized in the text?'
                question2.attrib['answer'] = ''

                question3 = ET.SubElement(questions, 'question3')
                question3.text = 'Does the text contain more information than in the modified tripleset?'
                question3.attrib['answer'] = 'y/n'

                question4 = ET.SubElement(questions, 'question4')
                question4.text = 'Did you find any mistake involving the references?'
                question4.attrib['answer'] = 'y/n'

                question5 = ET.SubElement(questions, 'question5')
                question5.text = 'Did you find any mistake involving the verbs? (e.g., The boy "play" soccer. instead of "plays")'
                question5.attrib['answer'] = 'y/n'

                question6 = ET.SubElement(questions, 'question6')
                question6.text = 'Did you find any mistake involving the determiners? (e.g., "An" boy.)'
                question6.attrib['answer'] = 'y/n'

                question7 = ET.SubElement(questions, 'question7')
                question7.text = 'What is the fluency of the text in a scale of 1-7?'
                question7.attrib['answer'] = ''

                question8 = ET.SubElement(questions, 'question8')
                question8.text = 'What is the semantics of the text in a scale of 1-7?'
                question8.attrib['answer'] = ''

        rough_string = ET.tostring(tree.getroot(), encoding='utf-8', method='xml')
        rough_string = re.sub(">\n[\n|\t| ]+<", '><', rough_string.decode('utf-8'))
        xml = minidom.parseString(rough_string).toprettyxml(indent="   ")

        with open(write_file, 'wb') as f:
            f.write(xml.encode('utf-8'))

if __name__ == '__main__':
    path = 'questionaire'
    if not os.path.exists(path):
        os.mkdir(path)

    eval = HumanEval()

    # End-to-End Transformer (Model 1)
    with open('pipeline/transformer/test.structing') as f:
        structing = f.read().split('\n')
    write_file = os.path.join(path, 'model1.xml')
    eval.qualitative_analysis(tag='e2etransformer', structs=structing, write_file=write_file)

    # End-to-End RNN (Model 2)
    with open('pipeline/rnn/test.structing') as f:
        structing = f.read().split('\n')
    write_file = os.path.join(path, 'model2.xml')
    eval.qualitative_analysis(tag='e2ernn', structs=structing, write_file=write_file)

    # Pipeline Transformer (Model 3)
    with open('pipeline/transformer/test.structing') as f:
        structing = f.read().split('\n')
    write_file = os.path.join(path, 'model3.xml')
    eval.qualitative_analysis(tag='transformer', structs=structing, write_file=write_file)

    # Pipeline RNN (Model 4)
    with open('pipeline/rnn/test.structing') as f:
        structing = f.read().split('\n')
    write_file = os.path.join(path, 'model4.xml')
    eval.qualitative_analysis(tag='rnn', structs=structing, write_file=write_file)

    # Pipeline Rand (Model 5)
    with open('pipeline/rand/test.structing') as f:
        structing = f.read().split('\n')
    write_file = os.path.join(path, 'model5.xml')
    eval.qualitative_analysis(tag='rand', structs=structing, write_file=write_file)

    # Pipeline Major (Model 6)
    with open('pipeline/major/test.structing') as f:
        structing = f.read().split('\n')
    write_file = os.path.join(path, 'model6.xml')
    eval.qualitative_analysis(tag='major', structs=structing, write_file=write_file)

    # Melbourne (Model 7)
    with open('pipeline/transformer/test.structing') as f:
        structing = f.read().split('\n')
    write_file = os.path.join(path, 'model7.xml')
    eval.qualitative_analysis(tag='melbourne', structs=structing, write_file=write_file)

    # UPF Forge (Model 8)
    with open('pipeline/transformer/test.structing') as f:
        structing = f.read().split('\n')
    write_file = os.path.join(path, 'model8.xml')
    eval.qualitative_analysis(tag='upfforge', structs=structing, write_file=write_file)

    # Original (Model 9)
    with open('pipeline/transformer/test.structing') as f:
        structing = f.read().split('\n')
    write_file = os.path.join(path, 'model9.xml')
    eval.qualitative_analysis(tag='original', structs=structing, write_file=write_file)