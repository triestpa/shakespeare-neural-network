'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = './textdatasets/tinyshakesepare.txt'
text = open(path).read().lower()

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Load model...')
model = load_model('shakes-iter-10.h5')

seed_string="this is a normal sample"
# seed_string="b"
print ("seed string -->", seed_string)
print ('The generated text is')
sys.stdout.write(seed_string),
#x=np.zeros((1, len(seed_string), len(chars)))
for i in range(10000):
    x=np.zeros((1, len(seed_string), len(chars)))
    for t, char in enumerate(seed_string):
        x[0, t, char_indices[char]] = 1.
    preds = model.predict(x, verbose=0)[0]
    #print (np.argmax(preds[7]))
    next_index=np.argmax(preds[len(seed_string)-1])

    #next_index=np.argmax(preds[len(seed_string)-11])
    #print (preds.shape)
    #print (preds)
    #next_index = sample(preds, 1) #diversity is 1
    next_char = indices_char[next_index]
    seed_string = seed_string + next_char

    #print (seed_string)
    #print ('##############')
    #if i==40:
    #    print ('####')
    sys.stdout.write(next_char)
    sys.stdout.flush()