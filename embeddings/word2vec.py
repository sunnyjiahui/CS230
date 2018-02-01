import numpy as np
import collections
import gensim
import pickle
from settings import ROOT_DIR
''' the input for gensim word2vec is a list of words, cannot be sentences as the document_test is'''

dir = ROOT_DIR+'\\processed_data\\';

data = pickle.load(open(dir+'document_text.p', 'rb'));

model = gensim.models.Word2Vec(data, min_count=1)

print(model.similarity('stock', 'market'))