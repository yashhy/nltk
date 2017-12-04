from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import nltk

# Might need the below line once
# nltk.download('punkt')

corpusDir = 'own_corpus/'

newCorpus = PlaintextCorpusReader(corpusDir, '.*\.txt')

for file in sorted(newCorpus.fileids()):
    words = newCorpus.words(file)
    text = nltk.Text(words)
    print(text)
