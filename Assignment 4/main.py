import numpy as np
import pandas as pd
import nltk
from pypdf import PdfReader
import os
from scipy.stats import zscore
from scipy.spatial.distance import mahalanobis

nltk.download('punkt', quiet=True)

def NGramExtractor(text):
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]

    dict_uni, dict_bi, dict_tri, dict_quad = {}, {}, {}, {}

    for i in range(len(tokens) - 3):
        token1, token2, token3, token4 = tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]

        unigrams = token1
        bigrams = token1 + " " + token2
        trigrams = token1 + " " + token2 + " " + token3
        quadgrams = token1 + " " + token2 + " " + token3 + " " + token4

        dict_uni[unigrams] = dict_uni.get(unigrams, 0) + 1
        dict_bi[bigrams] = dict_bi.get(bigrams, 0) + 1
        dict_tri[trigrams] = dict_tri.get(trigrams, 0) + 1
        dict_quad[quadgrams] = dict_quad.get(quadgrams, 0) + 1

    for i in np.arange(len(tokens) - 3, len(tokens)):
        unigrams = tokens[i]
        dict_uni[unigrams] = dict_uni.get(unigrams, 0) + 1

    for i in np.arange(len(tokens) - 3, len(tokens) - 1):
        bigrams = tokens[i] + " " + tokens[i + 1]
        dict_bi[bigrams] = dict_bi.get(bigrams, 0) + 1

    trigram = tokens[-3] + " " + tokens[-2] + " " + tokens[-1]
    dict_tri[trigram] = dict_tri.get(trigram, 0) + 1

    return {1: dict_uni, 2: dict_bi, 3: dict_tri, 4: dict_quad}

def Extract_NGrams_From_File(my_file):
    with open(my_file, 'rb') as pdf_file:
        reader = PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text().lower()
    return NGramExtractor(text)

file_gram_map = {}
file_list = os.listdir(r"D:\Programming\LAB\TY\Machine Learning\Assignment 4\Files")

global_map = {1: {}, 2: {}, 3: {}, 4: {}}
document_count_map = {1: {}, 2: {}, 3: {}, 4: {}}

for file in file_list:
    my_file = r"D:/Programming/LAB/TY/Machine Learning/Assignment 4/Files/" + file
    file_gram_map[my_file] = Extract_NGrams_From_File(my_file)

    for i in range(1, 5):
        i_gram = file_gram_map[my_file][i]
        for key, count in i_gram.items():
            global_map[i][key] = global_map[i].get(key, 0) + count
            document_count_map[i][key] = document_count_map[i].get(key, 0) + 1

def build_ngram_df(n):
    document_count_series = pd.Series(document_count_map[n])
    gram_count_series = pd.Series(global_map[n])
    df = pd.DataFrame({
        "gram_count": gram_count_series,
        "document_count": document_count_series
    })
    df["Average_freq"] = df["gram_count"] / df["document_count"]
    return df

unigrams = build_ngram_df(1)
bigrams = build_ngram_df(2)
trigrams = build_ngram_df(3)
quadgrams = build_ngram_df(4)

# print(unigrams)
# print(bigrams)
# print(trigrams)
# print(quadgrams)




output_folder = "results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


unigrams.to_csv(os.path.join(output_folder, "unigrams.csv"), index=True)
bigrams.to_csv(os.path.join(output_folder, "bigrams.csv"), index=True)
trigrams.to_csv(os.path.join(output_folder, "trigrams.csv"), index=True)
quadgrams.to_csv(os.path.join(output_folder, "quadgrams.csv"), index=True)


# z score for unigrams, bigrams, trigrams, quadgrams
unigrams['z_score']  = zscore(unigrams['gram_count'])
filtered_unigrams    = unigrams[(unigrams['z_score'] > -3) & (unigrams['z_score'] < 3)]

bigrams['z_score']   = zscore(bigrams['gram_count'])
filtered_bigrams     = bigrams[(bigrams['z_score'] > -3) & (bigrams['z_score'] < 3)]

trigrams['z_score']  = zscore(trigrams['gram_count'])
filtered_trigrams    = trigrams[(trigrams['z_score'] > -3) & (trigrams['z_score'] < 3)]

quadgrams['z_score'] = zscore(quadgrams['gram_count'])
filtered_quadgrams   = quadgrams[(quadgrams['z_score'] > -3) & (quadgrams['z_score'] < 3)]


# IQR for unigrams, bigrams, trigrams, quadgrams
Q1_unigram = unigrams['gram_count'].quantile(0.25)
Q3_unigram = unigrams['gram_count'].quantile(0.75)
IQR_unigram = Q3_unigram - Q1_unigram

filtered_unigrams_iqr = unigrams[
    (unigrams['gram_count'] >= Q1_unigram - 1.5 * IQR_unigram) &
    (unigrams['gram_count'] <= Q3_unigram + 1.5 * IQR_unigram)
]

Q1_bigram = bigrams['gram_count'].quantile(0.25)
Q3_bigram = bigrams['gram_count'].quantile(0.75)
IQR_bigram = Q3_bigram - Q1_bigram

filtered_bigrams_iqr = bigrams[
    (bigrams['gram_count'] >= Q1_bigram - 1.5 * IQR_bigram) &
    (bigrams['gram_count'] <= Q3_bigram + 1.5 * IQR_bigram)
]

Q1_trigram = trigrams['gram_count'].quantile(0.25)
Q3_trigram = trigrams['gram_count'].quantile(0.75)
IQR_trigram = Q3_trigram - Q1_trigram

filtered_trigrams_iqr = trigrams[
    (trigrams['gram_count'] >= Q1_trigram - 1.5 * IQR_trigram) &
    (trigrams['gram_count'] <= Q3_trigram + 1.5 * IQR_trigram)
]

Q1_quadgram = quadgrams['gram_count'].quantile(0.25)
Q3_quadgram = quadgrams['gram_count'].quantile(0.75)
IQR_quadgram = Q3_quadgram - Q1_quadgram

filtered_quadgrams_iqr = quadgrams[
    (quadgrams['gram_count'] >= Q1_quadgram - 1.5 * IQR_quadgram) &
    (quadgrams['gram_count'] <= Q3_quadgram + 1.5 * IQR_quadgram)
]


# MAD for unigrams, bigrams, tirgrams, quadgrams
median_unigram = unigrams['gram_count'].median()
mad_unigram = (unigrams['gram_count'] - median_unigram).abs().median()

filtered_unigrams_mad = unigrams[
    ((unigrams['gram_count'] - median_unigram).abs() / mad_unigram) < 3
]

median_bigram = bigrams['gram_count'].median()
mad_bigram = (bigrams['gram_count'] - median_bigram).abs().median()

filtered_bigrams_mad = bigrams[
    ((bigrams['gram_count'] - median_bigram).abs() / mad_bigram) < 3
]

median_trigram = trigrams['gram_count'].median()
mad_trigram = (trigrams['gram_count'] - median_trigram).abs().median()

filtered_trigrams_mad = trigrams[
    ((trigrams['gram_count'] - median_trigram).abs() / mad_trigram) < 3
]

median_quadgram = quadgrams['gram_count'].median()
mad_quadgram = (quadgrams['gram_count'] - median_quadgram).abs().median()

filtered_quadgrams_mad = quadgrams[
    ((quadgrams['gram_count'] - median_quadgram).abs() / mad_quadgram) < 3
]

# Mahalnobis distance multivariate thing for assignment n grams
X_unigram = unigrams[['gram_count', 'document_count', 'Average_freq']].values
cov_matrix_unigram = np.cov(X_unigram.T)
inv_cov_matrix_unigram = np.linalg.inv(cov_matrix_unigram)
mean_vector_unigram = X_unigram.mean(axis=0)
mdist_unigram = np.array([mahalanobis(x, mean_vector_unigram, inv_cov_matrix_unigram) for x in X_unigram])
unigrams['mahalanobis'] = mdist_unigram
filtered_unigrams_md = unigrams[unigrams['mahalanobis'] < 3]

X_bigram = bigrams[['gram_count', 'document_count', 'Average_freq']].values
cov_matrix_bigram = np.cov(X_bigram.T)
inv_cov_matrix_bigram = np.linalg.inv(cov_matrix_bigram)
mean_vector_bigram = X_bigram.mean(axis=0)
mdist_bigram = np.array([mahalanobis(x, mean_vector_bigram, inv_cov_matrix_bigram) for x in X_bigram])
bigrams['mahalanobis'] = mdist_bigram
filtered_bigrams_md = bigrams[bigrams['mahalanobis'] < 3]

X_trigram = trigrams[['gram_count', 'document_count', 'Average_freq']].values
cov_matrix_trigram = np.cov(X_trigram.T)
inv_cov_matrix_trigram = np.linalg.inv(cov_matrix_trigram)
mean_vector_trigram = X_trigram.mean(axis=0)
mdist_trigram = np.array([mahalanobis(x, mean_vector_trigram, inv_cov_matrix_trigram) for x in X_trigram])
trigrams['mahalanobis'] = mdist_trigram
filtered_trigrams_md = trigrams[trigrams['mahalanobis'] < 3]

X_quadgram = quadgrams[['gram_count', 'document_count', 'Average_freq']].values
cov_matrix_quadgram = np.cov(X_quadgram.T)
inv_cov_matrix_quadgram = np.linalg.inv(cov_matrix_quadgram)
mean_vector_quadgram = X_quadgram.mean(axis=0)
mdist_quadgram = np.array([mahalanobis(x, mean_vector_quadgram, inv_cov_matrix_quadgram) for x in X_quadgram])
quadgrams['mahalanobis'] = mdist_quadgram
filtered_quadgrams_md = quadgrams[quadgrams['mahalanobis'] < 3]



filtered_unigrams.to_csv("results/unigrams_zscore.csv", index=False)
filtered_unigrams_iqr.to_csv("results/unigrams_iqr.csv", index=False)
filtered_unigrams_mad.to_csv("results/unigrams_mad.csv", index=False)
filtered_unigrams_md.to_csv("results/unigrams_mahalanobis.csv", index=False)

filtered_bigrams.to_csv("results/bigrams_zscore.csv", index=False)
filtered_bigrams_iqr.to_csv("results/bigrams_iqr.csv", index=False)
filtered_bigrams_mad.to_csv("results/bigrams_mad.csv", index=False)
filtered_bigrams_md.to_csv("results/bigrams_mahalanobis.csv", index=False)

filtered_trigrams.to_csv("results/trigrams_zscore.csv", index=False)
filtered_trigrams_iqr.to_csv("results/trigrams_iqr.csv", index=False)
filtered_trigrams_mad.to_csv("results/trigrams_mad.csv", index=False)
filtered_trigrams_md.to_csv("results/trigrams_mahalanobis.csv", index=False)

filtered_quadgrams.to_csv("results/quadgrams_zscore.csv", index=False)
filtered_quadgrams_iqr.to_csv("results/quadgrams_iqr.csv", index=False)
filtered_quadgrams_mad.to_csv("results/quadgrams_mad.csv", index=False)
filtered_quadgrams_md.to_csv("results/quadgrams_mahalanobis.csv", index=False)

