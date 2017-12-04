import nltk
import pickle

classifier_list = ['naive_bayes', 'MNB','BernoulliNB', 'LogisticRegression','SGDClassifier', 'LinearSVC', 'NuSVC']


def classifer_test_pickle(classify_list):
    testing_sets_f = open("test/testing_sets.pickle", "rb")
    testing_sets = pickle.load(testing_sets_f)
    testing_sets_f.close()

    # print(testing_sets)

    for classifierName in classify_list:
        classifier_f = open("classifier/" + classifierName + ".pickle", "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()
        print(classifierName + ": ", ((nltk.classify.accuracy(classifier, testing_sets)) * 100), '%')


classifer_test_pickle(classifier_list)
