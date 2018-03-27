from sklearn.datasets import fetch_20newsgroups
from gensim.models import word2vec
from bs4 import BeautifulSoup
import re
import nltk
import time

start = time.time()
model_path = "./models/word2vec_gensim"

news = fetch_20newsgroups(subset='all')
X, y = news.data, news.target

# print len(X)
# print X[0]
# print y[:9]

def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
    return sentences


sentences = []
for x in X:
    sentences.extend(news_to_sentences(x))

num_featrues = 300
min_word_count = 20
num_workers = 2
context = 5
downsampling = 1e-3
max_vocab = 1000000

model = word2vec.Word2Vec(
    sentences,
    workers=num_workers,
    size=num_featrues,
    min_count=min_word_count,
    window=context,
    sample=downsampling,
    max_vocab_size=max_vocab
)

model.init_sims(replace=True)

print model['morning']
print model.most_similar('morning', topn=3)
print model.most_similar(positive=['man', 'son'], negative=['woman'], topn=4)
print model.similarity('woman', 'man')
list1 = ['the', 'cat', 'is', 'walking', 'in', 'the', 'bedroom']
list2 = ['the', 'dog', 'was', 'running', 'across', 'the', 'kitchen']
print model.n_similarity(list1, list2)

# #---save the trained model---
# model.save(model_path)
#
# #---load the trained model---
# model = word2vec.Word2Vec.load(model_path)
