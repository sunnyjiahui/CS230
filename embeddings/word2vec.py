import numpy as np
import collections
import gensim
import pickle
from settings import ROOT_DIR
''' the input for gensim word2vec is a sequence of sentences as its input.
    Each sentence a list of words (utf8 strings):'''

dir = ROOT_DIR+'\\processed_data\\';

data = pickle.load(open(dir+'sentence_text.p', 'rb'));
document = data[1];

#compiled sentences
compiled_sentences = data[0];
for i in range(len(data)):
    compiled_sentences += data[i];

print(compiled_sentences)
model = gensim.models.Word2Vec(compiled_sentences, min_count=1)

print(model.similarity('Greece', 'bailout'))