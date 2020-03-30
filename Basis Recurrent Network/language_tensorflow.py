import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from util import poetry_data, init_weight
from sklearn.utils import shuffle
from variables import *

class BaseRNN(object):
    def __init__(self,D,M,V):
        self.D = D # dimensions
        self.M = M # hidden dimension
        self.V = V # poetry_data

        # initial weights
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)

        self.We = tf.Variable(We, dtype=tf.float32)
        self.Wx = tf.Variable(Wx, dtype=tf.float32)
        self.Wh = tf.Variable(Wh, dtype=tf.float32)
        self.bh = tf.Variable(bh, dtype=tf.float32)
        self.h0 = tf.Variable(h0, dtype=tf.float32)
        self.Wo = tf.Variable(Wo, dtype=tf.float32)
        self.bo = tf.Variable(bo, dtype=tf.float32)

        self.parameters = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

    def recurrent_cell(self,tfX):
        D = self.D
        M = self.M
        V = self.V

        Xe = tf.nn.embedding_lookup(self.We, tfX)

        def recurrence_unit(ht_1,x_t):
            Xw = tf.matmul(tf.reshape(x_t,(1,D)), self.Wx)
            Hw = tf.matmul(tf.reshape(ht_1,(1,M)),self.Wh)
            h_t = tf.nn.relu(Hw + Xw + self.bh)
            h_t = tf.reshape(h_t, (M,))  # ht has shape (M,)
            return h_t

        h = tf.scan(  # h has shape (n,M) where n is the number of words in a sentence
            fn=recurrence_unit,
            elems=Xe,
            initializer=self.h0,
        )

        h = tf.reshape(h, (-1, M))

        log_Ys = tf.matmul(h,self.Wo) + self.bo # has shape (n,V)
        prediction = tf.argmax(log_Ys, axis=1)
        self.output_probs = tf.nn.softmax(log_Ys)
        self.predict = prediction

        return h

    def language_model(self, tfX,tfY, learning_rate = learning_rate):
        self.optimizer = tf.optimizers.Adam(learning_rate)
        with tf.GradientTape() as tape:

            weights = tf.transpose(self.Wo, [1,0])
            biases = self.bo

            h = self.recurrent_cell(tfX)

            labels = tf.reshape(tfY, (-1, 1))
            self.current_loss = tf.reduce_mean(
                                tf.nn.sampled_softmax_loss(
                                                weights=weights,
                                                biases=biases,
                                                labels=labels,
                                                inputs=h,
                                                num_sampled=50, # number of negative samples
                                                num_classes=self.V))

        gradients = tape.gradient(self.current_loss, self.parameters)
        self.optimizer.apply_gradients(zip(gradients, self.parameters))

    def fit(self, X, epochs=num_epochs):
        total_loss = []
        print("Started training .....")
        n_total = sum((len(sentence)+1) for sentence in X)
        for epoch in range(epochs):
            epoch_loss = 0
            X = shuffle(X)
            n_correct = 0
            for i,x in enumerate(X):
                input_sequence = [0] + x
                output_sequence = x + [1]

                self.language_model(input_sequence, output_sequence)
                for pj, xj in zip(self.predict, output_sequence):
                    if pj == xj:
                        n_correct += 1
                epoch_loss += float(self.current_loss.numpy())

            epoch_acc = (float(n_correct)/n_total)
            print("epoch:", epoch, "loss:", epoch_loss, "correct rate:", epoch_acc)
            total_loss.append(epoch_loss)
        plt.plot(total_loss)
        plt.show()

    def save_weights(self):
        weight_dict = {
                'We' : self.We,
                'Wx' : self.Wx,
                'Wh' : self.Wh,
                'bh' : self.bh,
                'h0' : self.h0,
                'Wo' : self.Wo,
                'bo' : self.bo,
                }
        outfile = open(weight_path,'wb')
        pickle.dump(weight_dict,outfile)
        outfile.close()
        print("Weights are Saved !!!")

    def load_weights(self):
        print("Weights are loading !!!")
        infile = open(weight_path,'rb')
        weight_dict = pickle.load(infile)
        infile.close()

        self.We = tf.Variable(weight_dict['We'], dtype=tf.float32)
        self.Wx = tf.Variable(weight_dict['Wx'], dtype=tf.float32)
        self.Wh = tf.Variable(weight_dict['Wh'], dtype=tf.float32)
        self.bh = tf.Variable(weight_dict['bh'], dtype=tf.float32)
        self.h0 = tf.Variable(weight_dict['h0'], dtype=tf.float32)
        self.Wo = tf.Variable(weight_dict['Wo'], dtype=tf.float32)
        self.bo = tf.Variable(weight_dict['bo'], dtype=tf.float32)

    def predict_word(self,tfX):
        h = self.recurrent_cell(tfX)
        return self.output_probs

    def generate_text(self, word2idx, n_sentences=5):
        initial_distribution = BaseRNN.initial_word_distribution(word2idx)
        init = len(initial_distribution)
        idx2word = {v:k for k,v in word2idx.items()}

        X = [np.random.choice(init, p=initial_distribution)]
        print(idx2word[X[0]], end=" ")

        line = 0
        while line < n_sentences:
            log_probs = self.predict_word(X).numpy()[0]
            word_idx = np.random.choice(init, p=log_probs)
            X.append(word_idx)
            if word_idx > 1: # make sure not start or end
                word = idx2word[word_idx]
                print(word, end=" ")
            elif word_idx == 1:
                # end token
                line += 1
                print('')
                if line < 4:
                    X = [ np.random.choice(V, p=np.squeeze(initial_distribution)) ] # reset to start of line
                    print(idx2word[X[0]], end=" ")

    @staticmethod
    def initial_word_distribution(word2idx):
        V = len(word2idx)
        pi = np.zeros(V)
        for sentence in sentences:
            pi[sentence[0]] += 1
        pi /= pi.sum()
        return pi


if __name__ == "__main__":
    sentences,word2idx = poetry_data()
    V = len(word2idx)
    rnn = BaseRNN(50,50,V)
    if not os.path.exists(weight_path):
        rnn.fit(sentences)
        rnn.save_weights()
    else:
        rnn.load_weights()
    rnn.generate_text(word2idx)