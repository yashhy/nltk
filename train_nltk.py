import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = []

for category in movie_reviews.categories():
    for fileId in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileId)), category))

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featureSets = [(find_features(reviews), category) for (reviews, category) in documents]

training_sets = featureSets[:1500]
testing_sets = featureSets[1500:]


# Run this below chunk of code only one, later pickle it and enjoy :)

classifier = nltk.NaiveBayesClassifier.train(training_sets)
MNB_classifier = SklearnClassifier(MultinomialNB()).train(training_sets)
BernoulliNB_classifier = SklearnClassifier(BernoulliNB()).train(training_sets)
LogisticRegression_classifier = SklearnClassifier(
    LogisticRegression()).train(training_sets)
SGDClassifier_classifier = SklearnClassifier(
    SGDClassifier()).train(training_sets)
LinearSVC_classifier = SklearnClassifier(LinearSVC()).train(training_sets)
NuSVC_classifier = SklearnClassifier(NuSVC()).train(training_sets)

classifier_list = [('naive_bayes', classifier), ('MNB', MNB_classifier),
                   ('BernoulliNB', BernoulliNB_classifier),
                   ('LogisticRegression',
                    LogisticRegression_classifier), ('SGDClassifier',
                                                     SGDClassifier_classifier),
                   ('LinearSVC', LinearSVC_classifier), ('NuSVC',
                                                         NuSVC_classifier)]

def classifer_train_pickle(classify_list):
    for classifier in classify_list:
        classifier_f = open('classifier/' + classifier[0] + ".pickle", "wb")
        pickle.dump(classifier[1], classifier_f)
        classifier_f.close()


classifer_train_pickle(classifier_list)



save_testing_sets = open("test/testing_sets.pickle", "wb")
pickle.dump(testing_sets, save_testing_sets)
save_testing_sets.close()
