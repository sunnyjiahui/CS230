import nltk
import os
import json
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pickle
from settings import ROOT_DIR

dir = os.getcwd()
data_dir = ROOT_DIR+'\\financial_times\\FT-archive-2013\\';
out_dir = ROOT_DIR+'\\processed_data\\';
print(dir)
counter = 0;
document_data = list(); #stores all texts as single string
word_data = list(); #stores text as individual words found in the article

for file in os.listdir(data_dir):
    print(file)
    data = json.load(open(data_dir+file,  encoding="utf8"));
    #pprint(data)

    text = data['bodyXML']
    document_data.append(text);
    # create the transform


    counter+=1;
    if(counter> 100):
        break;

## save text_data
file = open(out_dir+'document_text.p', 'wb')
pickle.dump(document_data, file);


vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(document_data)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[1]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())

