import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from nltk.probability import FreqDist
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix


def plot_LSA(data, labels):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(data)
        lsa_scores = lsa.transform(data)
        colors = ['orange', 'blue']
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        orange_patch = matplotlib.patches.mpatches.Patch(color='orange', label='Not')
        blue_patch   = matplotlib.patches.mpatches.Patch(color='blue', label='Real')
        plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})
        plt.show()


def plot_wordcloud(word_list, max_words=150):
    plt.figure(figsize=(12, 8))
    word_cloud = WordCloud(background_color='black', max_font_size=80,
                           stopwords=set(STOPWORDS), max_words=max_words, scale=3)\
        .generate(" ".join(word_list))
    
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()


def plot_cm(y_true, y_pred, title, figsize=(5, 5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
