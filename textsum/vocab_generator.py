import sys
import operator
import pandas as pd
from collections import Counter
train_file = sys.argv[1]
eval_file = sys.argv[2]
out_file = sys.argv[3]
train_data = pd.read_csv(train_file,delimiter='\t',header=None)
valid_data = pd.read_csv(eval_file,delimiter='\t',header=None)
train_data.columns = ['junk1','junk2','title','sentence']
valid_data.columns = ['junk1','junk2','title','sentence']
all_text = list(train_data['title'].values)+list(train_data['sentence'].values)+list(valid_data['title'].values) + list(valid_data['sentence'].values)
vocab = Counter()
for text in all_text:
    vocab.update(text.lower().strip().split())
vocab.update({'<s>':10e5})
vocab.update({'</s>':10e5})
vocab.update({'<PAD>':5})
vocab.update({'</d>':10e5})
vocab.update({'<d>':10e5})
vocab.update({'<UNK>':10e5})

vocab = sorted(vocab.items(), key=operator.itemgetter(1),reverse=True)
fw = open(out_file, 'w')
for (key,value) in vocab:
    fw.write('{} {}\n'.format(key,value))
fw.close()

