import numpy as np
import pandas as pd
from heapq import nlargest
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from enum import Enum

new_line = '\n'

categories = ["positive", "negative"]

c_values = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
gamma_values = [-5, -4, -3, -2, -1, 0, 1]

c_lin_values = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
c_quad_values = [0.8, 0.9, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]

c_rbf_values = [0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
gamma_rbf_values = [-1.5, -1.4, -1.3, -1.2, -1.1, -1, -0.9, -0.8, -0.7, -0.6, -0.5]


class Sentiment(Enum):
    POS = 1
    NEG = 0

class Processing:

    label_data = None
    loaded_data = None
    text_list = None
    sentiment_list = None
    bag_of_words = None
    encoded_bag_of_words = None
    freq_words = None
    count_vectorizer = None
    tfidf_vectorizer = None
    sentiment = None

    def __init__(self, path, sent=None):
        self.path = path
        if sent != None:
            self.sentiment = Sentiment[sent].value

    def load_data(self):
        df = pd.read_csv(self.path)
        self.label_data = df["sentiment"]
        if self.sentiment != None:
            self.loaded_data = df[df["sentiment"] == self.sentiment]
        else:
            self.loaded_data = df

    def column_to_list(self, column):
        if column == "text":
            self.text_list = self.loaded_data[column].tolist()
        elif column == "sentiment":
            self.sentiment_list = self.loaded_data[column].tolist()

    def feature_extraction_count(self):
        self.count_vectorizer = CountVectorizer(lowercase=True)
        self.count_vectorizer.fit(self.text_list)
        self.bag_of_words = self.count_vectorizer.fit_transform(self.text_list)
        self.encoded_bag_of_words = self.bag_of_words.toarray()

    def top_frequent_words_count(self, n):
        sum_words = self.bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.count_vectorizer.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        self.freq_words = words_freq[:n]

        return self.freq_words

    def feature_extraction_tfidf(self):
        self.tfidf_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
        self.tfidf_vectorizer.fit(self.text_list)
        self.bag_of_words = self.tfidf_vectorizer.fit_transform(self.text_list)
        self.encoded_bag_of_words = self.bag_of_words.toarray()

    def top_frequent_words_tfidf(self, n):
        sum_words = self.bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.tfidf_vectorizer.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        self.freq_words = words_freq[:n]

        return self.freq_words


class Classification:

    bag_of_words = None
    encoded_bag_of_words = None
    label_data = None

    def __init__(self, bag_of_words, encoded_bag_of_words, label_data):
        self.bag_of_words = bag_of_words
        self.label_data = label_data
        self.encoded_bag_of_words = encoded_bag_of_words 

    def linear_svm(self, c_val=1.0):
        clf = svm.SVC(kernel='linear', C = c_val)
        clf.fit(self.bag_of_words, self.label_data)
        # Obtain accuracy on the train set
        y_hat = clf.predict(self.encoded_bag_of_words)
        acc = accuracy_score(y_hat, self.label_data)
        print(f"Accuracy on the train set (linear_svm): {acc:.7f}")
        numSVs_Arr = clf.n_support_
        numSVs = numSVs_Arr[0] + numSVs_Arr[1]
        return clf, acc, numSVs

    def quadratic_svm(self, c_val=1.0):
        clf = svm.SVC(kernel='poly', C = c_val)
        clf.fit(self.bag_of_words, self.label_data)
        # Obtain accuracy on the train set
        y_hat = clf.predict(self.encoded_bag_of_words)
        acc = accuracy_score(y_hat, self.label_data)
        print(f"Accuracy on the train set (quadratic_svm): {acc:.7f}")
        numSVs_Arr = clf.n_support_
        numSVs = numSVs_Arr[0] + numSVs_Arr[1]
        return clf, acc, numSVs

    def rbf_svm(self, c_val=1.0, gamma_val='scale'):
        clf = svm.SVC(kernel='rbf', C = c_val, gamma = gamma_val)
        clf.fit(self.bag_of_words, self.label_data)
        # Obtain accuracy on the train set
        y_hat = clf.predict(self.encoded_bag_of_words)
        acc = accuracy_score(y_hat, self.label_data)
        print(f"Accuracy on the train set (rbf_svm): {acc:.7f}")
        numSVs_Arr = clf.n_support_
        numSVs = numSVs_Arr[0] + numSVs_Arr[1]
        return clf, acc, numSVs



def main():


    # This section of code pre-processes our data and completes
    # section 0.a of the assignment which is to check the top
    # ten most frequent words with the respectice vectorizer.
    #
    # Part - 0: Preprocessing
    pos_processor = Processing("IA3-train.csv", "POS")
    pos_processor.load_data()
    pos_processor.column_to_list("text")
    pos_processor.feature_extraction_count()
    neg_processor = Processing("IA3-train.csv", "NEG")
    neg_processor.load_data()
    neg_processor.column_to_list("text")
    neg_processor.feature_extraction_count()
    top_pos_words_count = pos_processor.top_frequent_words_count(10)
    top_neg_words_count = neg_processor.top_frequent_words_count(10)
    print(f"The Top Positive words(count): {top_pos_words_count}")
    print(f"{new_line}")
    print(f"The Top Negative words(count): {top_neg_words_count}")
    print(f"{new_line}")
    pos_processor.feature_extraction_tfidf()
    neg_processor.feature_extraction_tfidf()
    top_pos_words_tfidf = pos_processor.top_frequent_words_tfidf(10)
    top_neg_words_tfidf = neg_processor.top_frequent_words_tfidf(10)
    print(f"{new_line}")
    print(f"The Top Positive words(tfidf): {top_pos_words_tfidf}")
    print(f"{new_line}")
    print(f"The Top Negative words(tfidf): {top_neg_words_tfidf}")


    train_processor = Processing("IA3-train.csv")
    train_processor.load_data()
    train_processor.column_to_list("text")
    train_processor.column_to_list("sentiment")
    train_processor.feature_extraction_tfidf()

    test_processor = Processing("IA3-dev.csv")
    test_processor.load_data()
    test_processor.column_to_list("text")
    test_processor.column_to_list("sentiment")

    trained_tfidf_vectorizer = train_processor.tfidf_vectorizer
    x_test = trained_tfidf_vectorizer.transform(test_processor.text_list)

    trained_classifier = Classification(train_processor.bag_of_words, train_processor.encoded_bag_of_words, train_processor.label_data)

    acc_linear_train = []
    acc_linear_test = []
    acc_quadratic_train = []
    acc_quadratic_test = []
    num_SVs_linear = []
    num_SVs_quadratic = []

    
    # Part 1 & Part 2: Linear & Quadratic SVM
    for val in c_values:
        linear_clf, acc_l_train, num_l_SVs = trained_classifier.linear_svm(pow(10, val))
        quadratic_clf, acc_q_train, num_q_SVs = trained_classifier.quadratic_svm(pow(10, val))
        acc_linear_train.append(acc_l_train)
        acc_quadratic_train.append(acc_q_train)
        num_SVs_linear.append(num_l_SVs)
        num_SVs_quadratic.append(num_q_SVs)
        ylinear_predict = linear_clf.predict(x_test)
        yquadratic_predict = quadratic_clf.predict(x_test)
        acc_linear = accuracy_score(ylinear_predict, test_processor.label_data)
        acc_linear_test.append(acc_linear)
        acc_quadratic = accuracy_score(yquadratic_predict, test_processor.label_data)
        acc_quadratic_test.append(acc_quadratic)
        
        print(f"Accuracy on the linear-svm test set with C value 10^{val}: {acc_linear:.7f}")
        print(f"Accuracy on the quadratic-svm test set with C value 10^{val}: {acc_quadratic:.7f}")
        print(f"{new_line}")
        print("num of SVs for linear SVM is", num_l_SVs)
        print("num of SVs for quadratic SVM is", num_q_SVs)


    # plot two lines - Linear SVM
    plt.plot(c_values, acc_linear_train, 'o-b')
    plt.plot(c_values, acc_linear_test, 'o-g')
    # set axis titles
    plt.xlabel("Hyperparameter - C")
    plt.ylabel("Accuracy")
    # set chart title
    plt.title("Linear SVM")
    # legend
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.show()
    
    
    # plot two lines - Quadratic SVM
    plt.plot(c_values, acc_quadratic_train, 'o-b')
    plt.plot(c_values, acc_quadratic_test, 'o-g')
    # set axis titles
    plt.xlabel("Hyperparameter - C")
    plt.ylabel("Accuracy")
    # set chart title
    plt.title("Quadratic SVM")
    # legend
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.show()
    plt.close()

    # plot number of SVs v.s. c value - Linear SVM
    plt.plot(c_values, num_SVs_linear)
    # set axis titles
    plt.xlabel("Hyperparameter - C")
    plt.ylabel("Number of support vectors")
    # set chart title
    plt.title("Linear_support vectors v.s. c")

    plt.savefig('numSV_l.png')
    plt.close()

    # plot number of SVs v.s. c value - Quadratic SVM
    plt.plot(c_values, num_SVs_quadratic)
    # set axis titles
    plt.xlabel("Hyperparameter - C")
    plt.ylabel("Number of support vectors")
    # set chart title
    plt.title("Quadratic_support vectors v.s. c")

    plt.savefig('numSV_q.png')
    plt.close()    
    
    # Uncomment the below code(2 for loops) when trying to test for the boundary of best c
    # Linear and Quadratic - boundary search range
    """
    # Test on boundary of the search range - Linear SVM
    for val in c_lin_values:
        linear_clf, acc_l_train = trained_classifier.linear_svm(pow(10, val))
        ylinear_predict = linear_clf.predict(x_test)
        acc_linear = accuracy_score(ylinear_predict, test_processor.label_data)
        print(f"Accuracy on the linear-svm test set with C value 10^{val}: {acc_linear:.7f}")
        print(f"{new_line}")
        
    # Test on boundary of the search range - Quadratic SVM   
    for val in c_quad_values:
        quadratic_clf, acc_q_train = trained_classifier.quadratic_svm(pow(10, val))
        yquadratic_predict = quadratic_clf.predict(x_test)
        acc_quadratic = accuracy_score(yquadratic_predict, test_processor.label_data)
        print(f"Accuracy on the quadratic-svm test set with C value 10^{val}: {acc_quadratic:.7f}")
        print(f"{new_line}")
    
    """
    
    # Part 3 - RBF Kernel SVM
    acc_rbf_train = []
    acc_rbf_test = []
    num_SVs_rbf_fix_Gamma = []
    num_SVs_rbf_fix_c = []
    # Loop through all possible c and gamma value combinations    
    for val in c_values:
        for gamma in gamma_values:
            rbf_clf, acc_r_train, num_r_SVs = trained_classifier.rbf_svm(pow(10, val), pow(10, gamma))
            acc_rbf_train.append(acc_r_train)
            if gamma == -1:
                num_SVs_rbf_fix_Gamma.append(num_r_SVs)
            if val == 1:
                num_SVs_rbf_fix_c.append(num_r_SVs)
            yrbf_predict = rbf_clf.predict(x_test)
            acc_rbf = accuracy_score(yrbf_predict, test_processor.label_data)
            acc_rbf_test.append(acc_rbf)
            print(f"Accuracy on the rbf-svm test set with C value 10^{val} and gamma value {gamma}: {acc_rbf:.7f}")
            print(f"{new_line}")

    # plot number of SVs v.s. c value - rbf SVM --- fix gamma at 0.1
    plt.plot(c_values, num_SVs_rbf_fix_Gamma)
    # set axis titles
    plt.xlabel("Hyperparameter - C")
    plt.ylabel("Number of support vectors")
    # set chart title
    plt.title("rbf_support vectors v.s. c")

    plt.savefig('numSV_r_fix_gamma.png')
    plt.close() 

    # plot number of SVs v.s. gamma value - rbf SVM --- fix c at 10
    plt.plot(gamma_values, num_SVs_rbf_fix_c)
    # set axis titles
    plt.xlabel("Hyperparameter - Gamma")
    plt.ylabel("Number of support vectors")
    # set chart title
    plt.title("rbf_support vectors v.s. gamma")

    plt.savefig('numSV_r_fix_c.png')
    plt.close() 

    # save the train, test for all c and gamma combinations into a csv file
    i = 0
    with open('rbf_result.csv', 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(['c-value', 'gamma-value', 'train_accuracy', 'test_accuracy'])
        for val in c_values:
            for gamma in gamma_values:
                thewriter.writerow([val, gamma, acc_rbf_train[i], acc_rbf_test[i]])
                i = i + 1
    
    # open the csv file created as output for RBF Kernel SVM and plot heatmaps
    df = pd.read_csv("rbf_result.csv")
    df1 = df[['c-value', 'gamma-value', 'train_accuracy']]
    df2 = df[['c-value', 'gamma-value', 'test_accuracy']]
    
    
    # Heatmap plot - Training Accuracy
    heatmap1_data = pd.pivot_table(df1, values='train_accuracy', index='gamma-value', columns='c-value')
    fig, ax = plt.subplots(figsize=(12,7))
    title = "Heatmap - Training Accuracy"
    plt.title(title, fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    
    sns.heatmap(heatmap1_data,linewidths=0.30,ax=ax)
    plt.show()
    
    # Heatmap plot - Validation Accuracy
    heatmap2_data = pd.pivot_table(df2, values='test_accuracy', index='gamma-value', columns='c-value')
    fig, ax = plt.subplots(figsize=(12,7))
    title = "Heatmap - Validation Accuracy"
    plt.title(title, fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5,1.05])
    
    sns.heatmap(heatmap2_data,linewidths=0.30,ax=ax)
    plt.show()
    
    """
    # Uncomment the below code(for loop) when trying to test for the boundary of best c and gamma
    for val in c_rbf_values:
        for gamma in gamma_rbf_values:
            rbf_clf, acc_r_train = trained_classifier.rbf_svm(pow(10, val), pow(10, gamma))
            acc_rbf_train.append(acc_r_train)
            yrbf_predict = rbf_clf.predict(x_test)
            acc_rbf = accuracy_score(yrbf_predict, test_processor.label_data)
            acc_rbf_test.append(acc_rbf)
            print(f"Accuracy on the rbf-svm test set with C value 10^{val} and gamma value {gamma}: {acc_rbf:.7f}")
            print(f"{new_line}")
    """
    

if __name__ == '__main__':
    main()