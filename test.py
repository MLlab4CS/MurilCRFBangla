from flair.data import Sentence
from flair.models import SequenceTagger

from tqdm import tqdm
import json

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from seqeval.metrics import classification_report

columns = {0: 'text', 3: 'ner'}
corpus: Corpus = ColumnCorpus('data/', columns,
                              train_file='bn_train.txt',
                              dev_file='bn_dev.txt',
                              test_file='bn_test.txt'
                              )

# load the model you trained
model_mean = SequenceTagger.load('/model/best-model.pt')

result_mean = model_mean.evaluate(corpus.test, gold_label_type='ner',mini_batch_size=4, out_path=f"/model/test.tsv")
print(result_mean)