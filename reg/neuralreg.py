__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 25/11/2017
Description:
    NeuralREG+CAtt model concatenating the attention contexts from pre- and pos-contexts

    Based on https://github.com/clab/dynet/blob/master/examples/sequence-to-sequence/attention.py

    Attention()
        :param config
            LSTM_NUM_OF_LAYERS: number of LSTM layers
            EMBEDDINGS_SIZE: embedding dimensions
            STATE_SIZE: dimension of decoding output
            ATTENTION_SIZE: dimension of attention representations
            DROPOUT: dropout probabilities on the encoder and decoder LSTMs
            CHARACTER: character- (True) or word-based decoder
            GENERATION: max output limit
            BEAM_SIZE: beam search size

        train()
            :param fdir
                Directory to save best results and model

    PYTHON VERSION: 3

    DEPENDENCIES:
        Dynet: https://github.com/clab/dynet
        NumPy: http://www.numpy.org/

    UPDATE CONSTANTS:
        FDIR: directory to save results and trained models
"""

import dynet_config
dynet_config.set_gpu()
import dynet as dy
import json
import numpy as np
import os

class Config:
    def __init__(self, config):
        self.lstm_depth = config['LSTM_NUM_OF_LAYERS']
        self.embedding_dim = config['EMBEDDINGS_SIZE']
        self.state_dim = config['STATE_SIZE']
        self.attention_dim = config['ATTENTION_SIZE']
        self.dropout = config['DROPOUT']
        self.max_len = config['GENERATION']
        self.beam = config['BEAM_SIZE']
        self.batch = config['BATCH_SIZE']
        self.early_stop = config['EARLY_STOP']
        self.epochs = config['EPOCHS']

class NeuralREG():
    def __init__(self, config, path):
        self.path = path
        self.config = Config(config=config)

        self.EOS = "eos"
        self.vocab = json.load(open(os.path.join(self.path, 'vocab.json')))
        self.trainset = json.load(open(os.path.join(self.path, 'train.json')))
        self.devset = json.load(open(os.path.join(self.path, 'dev.json')))
        self.testset = json.load(open(os.path.join(self.path, 'test.json')))

        self.int2input = list(self.vocab['input'])
        self.input2int = {c:i for i, c in enumerate(self.vocab['input'])}

        self.int2output = list(self.vocab['output'])
        self.output2int = {c:i for i, c in enumerate(self.vocab['output'])}

        self.init()


    def init(self):
        dy.renew_cg()

        self.INPUT_VOCAB_SIZE = len(self.vocab['input'])
        self.OUTPUT_VOCAB_SIZE = len(self.vocab['output'])

        self.model = dy.Model()

        # ENCODERS
        self.encpre_fwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim, self.config.state_dim, self.model)
        self.encpre_bwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim, self.config.state_dim, self.model)
        self.encpre_fwd_lstm.set_dropout(self.config.dropout)
        self.encpre_bwd_lstm.set_dropout(self.config.dropout)

        self.encpos_fwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim, self.config.state_dim, self.model)
        self.encpos_bwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim, self.config.state_dim, self.model)
        self.encpos_fwd_lstm.set_dropout(self.config.dropout)
        self.encpos_bwd_lstm.set_dropout(self.config.dropout)

        # DECODER
        self.dec_lstm = dy.LSTMBuilder(self.config.lstm_depth, (self.config.state_dim*4)+(self.config.embedding_dim*2), self.config.state_dim, self.model)
        self.dec_lstm.set_dropout(self.config.dropout)

        # EMBEDDINGS
        self.input_lookup = self.model.add_lookup_parameters((self.INPUT_VOCAB_SIZE, self.config.embedding_dim))
        self.output_lookup = self.model.add_lookup_parameters((self.OUTPUT_VOCAB_SIZE, self.config.embedding_dim))

        # ATTENTION
        self.attention_w1_pre = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * 2))
        self.attention_w2_pre = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * self.config.lstm_depth * 2))
        self.attention_v_pre = self.model.add_parameters((1, self.config.attention_dim))

        self.attention_w1_pos = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * 2))
        self.attention_w2_pos = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * self.config.lstm_depth * 2))
        self.attention_v_pos = self.model.add_parameters((1, self.config.attention_dim))

        # SOFTMAX
        self.decoder_w = self.model.add_parameters((self.OUTPUT_VOCAB_SIZE, self.config.state_dim))
        self.decoder_b = self.model.add_parameters((self.OUTPUT_VOCAB_SIZE))


    def embed_sentence(self, sentence):
        _sentence = list(sentence)
        sentence = []
        for w in _sentence:
            try:
                sentence.append(self.input2int[w])
            except:
                sentence.append(self.input2int[self.EOS])

        return [self.input_lookup[char] for char in sentence]


    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors


    def encode_sentence(self, enc_fwd_lstm, enc_bwd_lstm, sentence):
        sentence_rev = list(reversed(sentence))

        fwd_vectors = self.run_lstm(enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = self.run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

        return vectors


    def attend(self, h, state, w1dt, attention_w2, attention_v):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = attention_w2*dy.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(attention_v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        context = h * att_weights
        return context


    def decode(self, pre_encoded, pos_encoded, output, entity):
        output = list(output)
        output = [self.output2int[c] for c in output]

        h_pre = dy.concatenate_cols(pre_encoded)
        w1dt_pre = None

        h_pos = dy.concatenate_cols(pos_encoded)
        w1dt_pos = None

        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]
        entity_embedding = self.input_lookup[self.input2int[entity]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.config.state_dim*4), last_output_embeddings, entity_embedding]))
        loss = []

        for word in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt_pre = w1dt_pre or self.attention_w1_pre * h_pre
            w1dt_pos = w1dt_pos or self.attention_w1_pos * h_pos

            attention_pre = self.attend(h_pre, s, w1dt_pre, self.attention_w2_pre, self.attention_v_pre)
            attention_pos = self.attend(h_pos, s, w1dt_pos, self.attention_w2_pos, self.attention_v_pos)

            vector = dy.concatenate([attention_pre, attention_pos, last_output_embeddings, entity_embedding])
            s = s.add_input(vector)
            out_vector = self.decoder_w * s.output() + self.decoder_b
            probs = dy.softmax(out_vector)
            last_output_embeddings = self.output_lookup[word]
            loss.append(-dy.log(dy.pick(probs, word)))
        loss = dy.esum(loss)
        return loss


    def generate(self, pre_context, pos_context, entity):
        embedded = self.embed_sentence(pre_context)
        pre_encoded = self.encode_sentence(self.encpre_fwd_lstm, self.encpre_bwd_lstm, embedded)

        embedded = self.embed_sentence(pos_context)
        pos_encoded = self.encode_sentence(self.encpos_fwd_lstm, self.encpos_bwd_lstm, embedded)

        h_pre = dy.concatenate_cols(pre_encoded)
        w1dt_pre = None

        h_pos = dy.concatenate_cols(pos_encoded)
        w1dt_pos = None

        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]
        try:
            entity_embedding = self.input_lookup[self.input2int[entity]]
        except:
            entity_embedding = self.input_lookup[self.input2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.config.state_dim*4), last_output_embeddings, entity_embedding]))

        out = []
        count_EOS = 0
        for i in range(self.config.max_len):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt_pre = w1dt_pre or self.attention_w1_pre * h_pre
            w1dt_pos = w1dt_pos or self.attention_w1_pos * h_pos

            attention_pre = self.attend(h_pre, s, w1dt_pre, self.attention_w2_pre, self.attention_v_pre)
            attention_pos = self.attend(h_pos, s, w1dt_pos, self.attention_w2_pos, self.attention_v_pos)

            vector = dy.concatenate([attention_pre, attention_pos, last_output_embeddings, entity_embedding])
            s = s.add_input(vector)
            out_vector = self.decoder_w * s.output() + self.decoder_b
            probs = dy.softmax(out_vector).vec_value()
            next_word = probs.index(max(probs))
            last_output_embeddings = self.output_lookup[next_word]
            if self.int2output[next_word] == self.EOS:
                count_EOS += 1
                continue

            out.append(self.int2output[next_word])

        return out


    def __call__(self, pre_context, pos_context, entity, beam):
        dy.renew_cg()
        return self.beam_search(pre_context, pos_context, entity, beam)


    def beam_search(self, pre_context, pos_context, entity, beam):
        embedded = self.embed_sentence(pre_context)
        pre_encoded = self.encode_sentence(self.encpre_fwd_lstm, self.encpre_bwd_lstm, embedded)

        embedded = self.embed_sentence(pos_context)
        pos_encoded = self.encode_sentence(self.encpos_fwd_lstm, self.encpos_bwd_lstm, embedded)

        h_pre = dy.concatenate_cols(pre_encoded)
        w1dt_pre = None

        h_pos = dy.concatenate_cols(pos_encoded)
        w1dt_pos = None

        try:
            entity_embedding = self.input_lookup[self.input2int[entity]]
        except:
            entity_embedding = self.input_lookup[self.input2int[self.EOS]]
        last_output_embeddings = self.output_lookup[self.output2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.config.state_dim*4), last_output_embeddings, entity_embedding]))
        candidates = [{'sentence':[self.EOS], 'prob':0.0, 'count_EOS':0, 's':s}]
        outputs = []

        i = 0
        while i < self.config.max_len and len(outputs) < beam:
            new_candidates = []
            for candidate in candidates:
                if candidate['count_EOS'] == 2:
                    outputs.append(candidate)

                    if len(outputs) == beam: break
                else:
                    # w1dt can be computed and cached once for the entire decoding phase
                    w1dt_pre = w1dt_pre or self.attention_w1_pre * h_pre
                    w1dt_pos = w1dt_pos or self.attention_w1_pos * h_pos

                    attention_pre = self.attend(h_pre, candidate['s'], w1dt_pre, self.attention_w2_pre, self.attention_v_pre)
                    attention_pos = self.attend(h_pos, candidate['s'], w1dt_pos, self.attention_w2_pos, self.attention_v_pos)

                    last_output_embeddings = self.output_lookup[self.output2int[candidate['sentence'][-1]]]
                    vector = dy.concatenate([attention_pre, attention_pos, last_output_embeddings, entity_embedding])
                    s = candidate['s'].add_input(vector)
                    out_vector = self.decoder_w * s.output() + self.decoder_b
                    probs = dy.softmax(out_vector).vec_value()
                    next_words = [{'prob':e, 'index':probs.index(e)} for e in sorted(probs, reverse=True)[:beam]]

                    for next_word in next_words:
                        word = self.int2output[next_word['index']]

                        new_candidate = {
                            'sentence': candidate['sentence'] + [word],
                            'prob': candidate['prob'] + np.log(next_word['prob']),
                            'count_EOS': candidate['count_EOS'],
                            's':s
                        }

                        if word == self.EOS:
                            new_candidate['count_EOS'] += 1

                        new_candidates.append(new_candidate)

            # # Length Normalization
            # alpha = 0.6
            # for candidate in new_candidates:
            #     length = len(candidate['sentence'])
            #     lp_y = ((5.0 + length)**alpha) / ((5.0+1.0)**alpha)
            #
            #     candidate['prob'] /= lp_y
            candidates = sorted(new_candidates, key=lambda x: x['prob'], reverse=True)[:beam]
            i += 1

        if len(outputs) == 0:
            outputs = candidates

        # Length Normalization
        alpha = 0.6
        for output in outputs:
            length = len(output['sentence'])
            lp_y = ((5.0 + length)**alpha) / ((5.0+1.0)**alpha)

            output['prob'] /= lp_y

        outputs = sorted(outputs, key=lambda x: x['prob'], reverse=True)
        return list(map(lambda x: x['sentence'], outputs))


    def get_loss(self, pre_context, pos_context, refex, entity):
        embedded = self.embed_sentence(pre_context)
        pre_encoded = self.encode_sentence(self.encpre_fwd_lstm, self.encpre_bwd_lstm, embedded)

        embedded = self.embed_sentence(pos_context)
        pos_encoded = self.encode_sentence(self.encpos_fwd_lstm, self.encpos_bwd_lstm, embedded)

        return self.decode(pre_encoded, pos_encoded, refex, entity)


    def write(self, fname, outputs):
        f = open(fname, 'w')
        for output in outputs:
            f.write(output[0])
            f.write('\n')

        f.close()


    def validate(self):
        results = []
        num, dem = 0.0, 0.0
        for i, devinst in enumerate(self.devset):
            pre_context = [self.EOS] + devinst['pre_context']
            pos_context = devinst['pos_context'] + [self.EOS]
            entity = devinst['entity']
            if self.config.beam == 1:
                outputs = [self.generate(pre_context, pos_context, entity)]
            else:
                outputs = self.beam_search(pre_context, pos_context, entity, self.config.beam)

            delimiter = ' '
            for j, output in enumerate(outputs):
                outputs[j] = delimiter.join(output).replace('eos', '').strip()
            refex = delimiter.join(devinst['refex']).replace('eos', '').strip()

            best_candidate = outputs[0]
            if refex == best_candidate:
                num += 1
            dem += 1

            if i < 20:
                print ("Refex: ", refex, "\t Output: ", best_candidate)
                print(10 * '-')

            results.append(outputs)

            if i % self.config.batch == 0:
                dy.renew_cg()

        return results, num, dem


    def train(self):
        trainer = dy.AdadeltaTrainer(self.model)

        log = []
        best_acc, repeat = 0.0, 0
        for epoch in range(self.config.epochs):
            dy.renew_cg()
            losses = []
            closs = 0.0
            for i, traininst in enumerate(self.trainset):
                pre_context = [self.EOS] + traininst['pre_context']
                pos_context = traininst['pos_context'] + [self.EOS]
                refex = [self.EOS] + traininst['refex'] + [self.EOS]
                entity = traininst['entity']
                loss = self.get_loss(pre_context, pos_context, refex, entity)
                losses.append(loss)

                if len(losses) == self.config.batch:
                    loss = dy.esum(losses)
                    closs += loss.value()
                    loss.backward()
                    trainer.update()
                    dy.renew_cg()

                    print("Epoch: {0} \t Loss: {1} \t Progress: {2}".format(epoch, (closs / self.config.batch), round(i / len(self.trainset), 2)), end='       \r')
                    losses = []
                    closs = 0.0

            outputs, num, dem = self.validate()
            acc = float(num) / dem
            log.append(acc)

            print("Dev acc: {0} \t Best acc: {1}".format(round(acc, 2), best_acc))

            # Saving the model with best accuracy
            if best_acc == 0.0 or acc > best_acc:
                best_acc = acc

                fname = 'dev_best.txt'
                self.write(os.path.join(self.path, fname), outputs)

                fname = 'best_model.dy'
                self.model.save(os.path.join(self.path, fname))

                repeat = 0
            else:
                repeat += 1

            # In case the accuracy does not increase in 20 epochs, break the process
            if repeat == self.config.early_stop:
                break

        json.dump(log, open(os.path.join(self.path, 'log.json'), 'w'))


    def evaluate(self, procset, write_path):
        results = []
        num, dem = 0.0, 0.0
        for i, devinst in enumerate(procset):
            print('Progress: ', round(i / len(procset), 2), end='\r')
            pre_context = [self.EOS] + devinst['pre_context']
            pos_context = devinst['pos_context'] + [self.EOS]
            entity = devinst['entity']
            if self.config.beam == 1:
                outputs = [self.generate(pre_context, pos_context, entity)]
            else:
                outputs = self.beam_search(pre_context, pos_context, entity, self.config.beam)

            delimiter = ' '
            for j, output in enumerate(outputs):
                outputs[j] = delimiter.join(output).replace('eos', '').strip()
            refex = delimiter.join(devinst['refex']).replace('eos', '').strip()

            best_candidate = outputs[0]
            if refex == best_candidate:
                num += 1
            dem += 1

            results.append(outputs)

            if i % self.config.batch == 0:
                dy.renew_cg()

        self.write(write_path, results)


    def save(self, path):
        self.model.save(path)


    def populate(self, path):
        self.model.populate(path)


if __name__ == '__main__':
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
        'EARLY_STOP': 10
    }

    # path = '/roaming/tcastrof/emnlp2019/reg'
    # for i in range(3):
    #     neuralreg = NeuralREG(path=path, config=config)
    #     neuralreg.train()
    #     os.rename(os.path.join(path, 'best_model.dy'), os.path.join(path, 'model'+str(i+1)+'.dy'))

    path = '.'
    neuralreg = NeuralREG(path=path, config=config)
    neuralreg.populate(os.path.join(path, 'model1.dy'))
    neuralreg.evaluate(neuralreg.devset, os.path.join(path, 'dev.out'))
    neuralreg.evaluate(neuralreg.testset, os.path.join(path, 'test.out'))
