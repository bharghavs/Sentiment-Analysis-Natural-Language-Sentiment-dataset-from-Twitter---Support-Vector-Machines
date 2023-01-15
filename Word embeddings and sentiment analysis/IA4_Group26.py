import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mtl
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import contingency_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from enum import Enum

# IA4: Group-26: Cheng, Bharath, Bharghav

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Loads GloVe embeddings from a designated file location. 
#
# Invoked via:
# ge = GloVe_Embedder(path_to_embeddings)
#
# Embed single word via:
# embed = ge.embed_str(word)
#
# Embed a list of words via:
# embeds = ge.embed_list(word_list)
#
# Find nearest neighbors via:
# ge.find_k_nearest(word, k)
#
# Save vocabulary to file via:
# ge.save_to_file(path_to_file)

class GloVe_Embedder:
    def __init__(self, path):
        self.embedding_dict = {}
        self.embedding_array = []
        self.unk_emb = 0
        # Adapted from https://stackoverflow.com/questions/37793118/load-pretrained-GloVe-vectors-in-python
        with open(path,'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.embedding_dict[word] = embedding
                self.embedding_array.append(embedding.tolist())
        self.embedding_array = np.array(self.embedding_array)
        self.embedding_dim = len(self.embedding_array[0])
        self.vocab_size = len(self.embedding_array)
        self.unk_emb = np.zeros(self.embedding_dim)

    # Check if the provided embedding is the unknown embedding.
    def is_unk_embed(self, embed):
        return np.sum((embed - self.unk_emb) ** 2) < 1e-7
    
    # Check if the provided string is in the vocabulary.
    def token_in_vocab(self, x):
        if x in self.embedding_dict and not self.is_unk_embed(self.embedding_dict[x]):
            return True
        return False

    # Returns the embedding for a single string and prints a warning if
    # the string is unknown to the vocabulary.
    # 
    # If indicate_unk is set to True, the return type will be a tuple of 
    # (numpy array, bool) with the bool indicating whether the returned 
    # embedding is the unknown embedding.
    #
    # If warn_unk is set to False, the method will no longer print warnings
    # when used on unknown strings.
    def embed_str(self, x, indicate_unk = False, warn_unk = True):
        if self.token_in_vocab(x):
            if indicate_unk:
                return (self.embedding_dict[x], False)
            else:
                return self.embedding_dict[x]
        else:
            if warn_unk:
                    print("Warning: provided word is not part of the vocabulary!")
            if indicate_unk:
                return (self.unk_emb, True)
            else:
                return self.unk_emb

    # Returns an array containing the embeddings of each vocabulary token in the provided list.
    #
    # If include_unk is set to False, the returned list will not include any unknown embeddings.
    def embed_list(self, x, include_unk = True):
        if include_unk:
            embeds = [self.embed_str(word, warn_unk = False).tolist() for word in x]
        else:
            embeds_with_unk = [self.embed_str(word, indicate_unk=True, warn_unk = False) for word in x]
            embeds = [e[0].tolist() for e in embeds_with_unk if not e[1]]
            if len(embeds) == 0:
                print("No known words in input:" + str(x))
                embeds = [self.unk_emb.tolist()]
        return np.array(embeds)
    
    # Finds the vocab words associated with the k nearest embeddings of the provided word. 
    # Can also accept an embedding vector in place of a string word.
    # Return type is a nested list where each entry is a word in the vocab followed by its 
    # distance from whatever word was provided as an argument.
    def find_k_nearest(self, word, k, warn_about_unks = True):
        if type(word) == str:
            word_embedding, is_unk = self.embed_str(word, indicate_unk = True)
        else:
            word_embedding = word
            is_unk = False
        if is_unk and warn_about_unks:
            print("Warning: provided word is not part of the vocabulary!")

        all_distances = np.sum((self.embedding_array - word_embedding) ** 2, axis = 1) ** 0.5
        distance_vocab_index = [[w, round(d, 5)] for w,d,i in zip(self.embedding_dict.keys(), all_distances, range(len(all_distances)))]
        distance_vocab_index = sorted(distance_vocab_index, key = lambda x: x[1], reverse = False)
        return distance_vocab_index[:k]

    def save_to_file(self, path):
        with open(path, 'w') as f:
            for k in self.embedding_dict.keys():
                embedding_str = " ".join([str(round(s, 5)) for s in self.embedding_dict[k].tolist()])
                string = k + " " + embedding_str
                f.write(string + "\n")


# part-1: Using word embeddings to improve classification
# Find thirty words for each of the five seed words
# data format: [word, distance to the seed word]--- see how the 150 words are represented in our shared google doc
# ge = GloVe_Embedder('/Users/chengzhen/Box/OSUstudy/courses/2022fall/AI534/IA4/GloVe_Embedder_data.txt')
ge = GloVe_Embedder(Path('GloVe_Embedder_data.txt'))
words_flight = ge.find_k_nearest("flight", 30)
words_good = ge.find_k_nearest("good", 30)
words_terrible = ge.find_k_nearest("terrible", 30)
words_help = ge.find_k_nearest("help", 30)
words_late = ge.find_k_nearest("late", 30)

# add the 150 words together with their in 20-dimension from the raw embedder
# the dimension is finally 150*20 (150 words * 20 coordinates)
raw = []
for i in range(30):
    raw.append(ge.embed_str(words_flight[i][0]))

for i in range(30):
    raw.append(ge.embed_str(words_good[i][0]))

for i in range(30):
    raw.append(ge.embed_str(words_terrible[i][0]))

for i in range(30):
    raw.append(ge.embed_str(words_help[i][0]))

for i in range(30):
    raw.append(ge.embed_str(words_late[i][0]))

# refer to https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit_transform
raw = np.asarray(raw)
# set the PCA in 2-D
pca = PCA(n_components=2)
# fit the data
pca.fit(raw)
# Fit the model with raw data and apply the dimensionality reduction.
pca_post = pca.transform(raw)
# after doing PCA into 2-D, the result (pca_post) is in 150*2 (150 words * 2 coordinates)
for i in range(30):
    plt.scatter(pca_post[i][0], pca_post[i][1], c = 'b')
for i in range(30, 60):
    plt.scatter(pca_post[i][0], pca_post[i][1], c = 'r')
for i in range(60, 90):
    plt.scatter(pca_post[i][0], pca_post[i][1], c = 'g')
for i in range(90, 120):
    plt.scatter(pca_post[i][0], pca_post[i][1], c = 'c')
for i in range(120, 150):
    plt.scatter(pca_post[i][0], pca_post[i][1], c = 'k')

plt.title('PCA - Cluster Visualization')
seed_words = ['flight','good','terrible','help','late']
legend_elements = [mtl.Line2D([0], [0], marker = 'o', markerfacecolor='b',label='flight'),
                   mtl.Line2D([0], [0], marker = 'o', markerfacecolor='r',label='good'),
                   mtl.Line2D([0], [0], marker = 'o', markerfacecolor='g',label='terrible'),
                   mtl.Line2D([0], [0], marker = 'o', markerfacecolor='c',label='help'),
                   mtl.Line2D([0], [0], marker = 'o', markerfacecolor='k',label='late')]
plt.legend(handles=legend_elements, loc='upper right')
plt.show()
plt.close()


# TSNE: refer to https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
perplexity_value = [5, 20, 35, 50]
for val in perplexity_value:
    post_tsne = TSNE(n_components=2, perplexity=val).fit_transform(raw)
    for i in range(150):
        plt.scatter(post_tsne[i][0], post_tsne[i][1], c = 'k')
    plt.title("t-SNE with Perplexity value: " + str(val))
    plt.show()
    plt.close()

# Clustering - kmeans clustering
sse = []
labels_pred = []
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=10)
    kmeans.fit(raw)
    sse.append(kmeans.inertia_)
    labels_pred.append(kmeans.labels_)
    
plt.plot(range(2, 21), sse, color='green', ls='-', marker='o')
plt.xticks(range(2, 21))
plt.title("kmeans objective as a function of k")
plt.xlabel("k - num of clusters")
plt.ylabel("kmeans objective")
plt.show()
plt.close()

# Original seed word as ground truth labels
labels_truth = []
for i in range(len(seed_words)):
    for j in range(30):
        labels_truth.append(i)


# Purity
purity_score = []
for i in range(len(sse)):
    matrix = contingency_matrix(np.array(labels_truth), np.array(labels_pred[i]))
    purity_metric = (np.sum(np.amax(matrix, axis=0)) / np.sum(matrix))
    purity_score.append(purity_metric)

plt.plot(range(2, 21), purity_score, color='green', ls='-', marker='o')
plt.xticks(range(2, 21))
plt.title("Purity vs k")
plt.xlabel("k - num of clusters")
plt.ylabel("Purity Score")
plt.show()
plt.close()

# Adjusted rand Score
adj_rand_score = []
for i in range(len(sse)):
    adj_rand_score.append(adjusted_rand_score(labels_truth, labels_pred[i]))

plt.plot(range(2, 21), adj_rand_score, color='green', ls='-', marker='o')
plt.xticks(range(2, 21))
plt.title("Adjusted Rand Index vs k")
plt.xlabel("k - num of clusters")
plt.ylabel("Adjusted Rand Score")
plt.show()
plt.close()
        
# Normalized Mutual Information
norm_mut_info_score = []
for i in range(len(sse)):
    norm_mut_info_score.append(normalized_mutual_info_score(labels_truth, labels_pred[i]))

plt.plot(range(2, 21), norm_mut_info_score, color='green', ls='-', marker='o')
plt.xticks(range(2, 21))
plt.title("Normalized Mutual Information vs k")
plt.xlabel("k - num of clusters")
plt.ylabel("Normalized Mutual Information score")
plt.show()
plt.close()



# Part 2: Using word embeddings to improve classification
# Built the below part-2 code from our IA-3 implementation
new_line = '\n'

categories = ["positive", "negative"]


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


    def feature_extraction_tfidf(self):
        self.tfidf_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
        self.tfidf_vectorizer.fit(self.text_list)
        self.bag_of_words = self.tfidf_vectorizer.fit_transform(self.text_list)
        self.encoded_bag_of_words = self.bag_of_words.toarray()



class Classification:

    bag_of_words = None
    encoded_bag_of_words = None
    label_data = None

    def __init__(self, bag_of_words, encoded_bag_of_words, label_data):
        self.bag_of_words = bag_of_words
        self.label_data = label_data
        self.encoded_bag_of_words = encoded_bag_of_words 
        
    def lgbm(self):
        lgb = LGBMClassifier()
        lgbm_model = lgb.fit(self.bag_of_words, self.label_data)
        y_hat = lgbm_model.predict(self.encoded_bag_of_words)
        acc_lgbm = accuracy_score(y_hat,self.label_data)
        print(f"Accuracy on the train set (lgbm_train_tf): {acc_lgbm:.7f}")
        return lgbm_model, acc_lgbm

    def xgb(self):
        xg = XGBClassifier()
        xgb_model = xg.fit(self.bag_of_words, self.label_data)
        y_hat_xgb = xgb_model.predict(self.encoded_bag_of_words)
        acc_xgb = accuracy_score(y_hat_xgb,self.label_data)
        print(f"Accuracy on the train set (xgb_train_tf): {acc_xgb:.7f}")
        return xgb_model, acc_xgb
    
    def random(self):
        rfc = RandomForestClassifier()
        rfc_model = rfc.fit(self.bag_of_words, self.label_data)
        y_hat_rfc = rfc_model.predict(self.encoded_bag_of_words)
        acc_rfc = accuracy_score(y_hat_rfc,self.label_data)
        print(f"Accuracy on the train set (rfc_train_tf): {acc_rfc:.7f}")
        return rfc_model, acc_rfc


def main():


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

    # xgboost
    xgboost_clf_tf, acc_xgb_train = trained_classifier.xgb()
    y_predict_tf = xgboost_clf_tf.predict(x_test)
    acc_xgb = accuracy_score(y_predict_tf, test_processor.label_data)
    print(f"Accuracy on the test set (xgb_test_tf): {acc_xgb:.7f}")

    # light gbm
    lgbm_clf_tf, acc_lgbm_train = trained_classifier.lgbm()
    y_predict_tf = lgbm_clf_tf.predict(x_test)
    acc_lgbm = accuracy_score(y_predict_tf, test_processor.label_data)
    print(f"Accuracy on the test set (lgbm_test_tf): {acc_lgbm:.7f}")
    
    # random forest
    random_clf_tf, acc_random_train = trained_classifier.random()
    y_predict_tf = random_clf_tf.predict(x_test)
    acc_random = accuracy_score(y_predict_tf, test_processor.label_data)
    print(f"Accuracy on the test set (rfc_test_tf): {acc_random:.7f}")
    

if __name__ == '__main__':
    main()