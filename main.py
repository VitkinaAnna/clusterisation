import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nltk
import re
import torch
import uuid
import os
import csv
import pickle

from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


from gensim.models import LdaModel, TfidfModel
from gensim.corpora import Dictionary

from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')



initial_df = pd.read_csv('projects_proposals.csv')

initial_df.shape

initial_df.head(5)

titles = initial_df["title"]

newdf = initial_df.drop(["supervisor", "company"], axis='columns')

newdf.head(5)


newdf = newdf.reset_index()
newdf = newdf.drop(['index'], axis=1)

newdf = newdf.drop(['level_0'], axis=1)

newdf = newdf.drop(['Unnamed: 0'], axis=1)


newdf.shape



# Preprocessing

def remove_punctuation_and_digits_df(df):
    tokenizer = RegexpTokenizer(r'\w+')
    for index, row in df.iterrows():
        preprocessed_string = " ".join(tokenizer.tokenize(row['abstract'])) # remove punctuaion
        preprocessed_string_without_digits = re.sub(r'\d+', '', preprocessed_string) # remove digits
        preprocessed_string_without_digits_lowercase = preprocessed_string_without_digits.lower()
        df.at[index,'abstract'] = preprocessed_string_without_digits_lowercase
    return df

preprocessed_df = remove_punctuation_and_digits_df(newdf)

string = preprocessed_df["abstract"][1]

# Remove stop words

def remove_stop_words_df(df):
    stop_words = set(stopwords.words('english'))
    for index, row in df.iterrows():
        word_tokens = word_tokenize(row['abstract'])
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        df.at[index,'abstract'] = " ".join(filtered_sentence)
    return df

preprocessed_stop_words_df = remove_stop_words_df(preprocessed_df)

summaries_for_bert = preprocessed_stop_words_df['abstract']

string = preprocessed_stop_words_df["abstract"][1]



# Apply POS tagging

# Extract adjectives
def get_adjectives(df, column_name):
    for index, row in newdf.iterrows():
        adjectives_list = []
        tokens = nltk.word_tokenize(row[column_name])
        tags = nltk.pos_tag(tokens)
        adjectives_list = [word for word, pos in tags if (pos == 'JJ')]
    return adjectives_list


adjectives_list = get_adjectives(preprocessed_stop_words_df, 'abstract')

# Leave only nouns and adjectives in text

def get_adjectives_and_nouns(df, column_name):
    for index, row in df.iterrows():
        tokens = nltk.word_tokenize(row[column_name])
        tags = nltk.pos_tag(tokens)
        nouns = [word for word, pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'JJ')]
        new_str = " ".join(nouns)
        df.at[index,column_name] = new_str
    return df


preprocessed_stop_words_adjectives_nouns_df = get_adjectives_and_nouns(preprocessed_stop_words_df, 'abstract')


# Lemmatization

def lemmatize_df(df, column_name):
    lemmatizer = WordNetLemmatizer()
    for index, row in preprocessed_stop_words_adjectives_nouns_df.iterrows():
        tokens = nltk.word_tokenize(row[column_name])
        lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
        new_str = " ".join(lemmatized_words)
        df.at[index,column_name] = new_str
    return df


preprocessed_stop_words_adjectives_nouns_lemmatized_df = lemmatize_df(preprocessed_stop_words_adjectives_nouns_df, "abstract")

# Get abstract column

def get_specific_column_from_df(df, column_name):
    column = []
    for index, row in df.iterrows():
        column.append(row[column_name].split())
    column = pd.Series(column)
    return column

abstracts = get_specific_column_from_df(preprocessed_stop_words_adjectives_nouns_lemmatized_df, "abstract")


# Get Bigrams and Trigrams

def get_bigrams_and_bigram_scores(column, frequency_filter):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_documents(column)
    finder.apply_freq_filter(frequency_filter)
    bigram_scores = finder.score_ngrams(bigram_measures.pmi)
    bigrams = [" ".join(list(bigram[0])) for bigram in bigram_scores]
    return bigram_scores, bigrams


frequency_filter = 30
bigram_scores, bigrams = get_bigrams_and_bigram_scores(abstracts, frequency_filter)

def get_trigrams_and_trigram_scores(column, frequency_filter):
  trigram_measures = nltk.collocations.TrigramAssocMeasures()
  finder_trigram = nltk.collocations.TrigramCollocationFinder.from_documents(column)
  finder_trigram.apply_freq_filter(frequency_filter)
  trigram_scores = finder_trigram.score_ngrams(trigram_measures.pmi)
  trigrams = [" ".join(list(trigram[0])) for trigram in trigram_scores]
  return trigrams, trigram_scores

frequency_filter = 10
trigrams, trigram_scores = get_trigrams_and_trigram_scores(abstracts, frequency_filter)


def get_trigrams(abstracts):
    trigrams = []
    whole_text = []
    for abstract in abstracts:
        whole_text = whole_text + abstract

    ngrams = Counter(nltk.ngrams(whole_text, 3))

    for ngram, freq in ngrams.most_common(11):
        print(ngram, freq)
        trigrams.append(" ".join(ngram))
    return trigrams


trigrams_another_method = get_trigrams(abstracts)



# Replacing N-grams with one word (e.g. machine learning -> machine_learning)

def replace_ngram(x, ngram_array):
    for gram in ngram_array:
        x = x.replace(gram, '_'.join(gram.split()))
    return x


def preprocess_ngram_df(series_column):
  tokenized_docs = series_column.apply(lambda x : " ".join(word for word in x))
  tokenized_docs = tokenized_docs.apply(lambda x: replace_ngram(x, bigrams))
  tokenized_docs = tokenized_docs.apply(lambda x: replace_ngram(x, trigrams))
  tokenized_docs = tokenized_docs.apply(lambda x: x.split())
  return tokenized_docs

tokenized_docs = preprocess_ngram_df(abstracts)

# Get vocabulary from words

def form_vocab(tokenized_docs):
    vocab = set()
    for i in tokenized_docs:
        for j in i:
            vocab.add(j)
    return vocab


vocab = form_vocab(tokenized_docs)

# Visualize word occurences

def visualize_word_occurencies(vocab, tokenized_docs):
    vocab_data = {i : [] for i in list(vocab)}
    word_occrencies_df = pd.DataFrame(vocab_data)

    for doc in tokenized_docs:
        word_occrencies_df = word_occrencies_df._append({i : doc.count(i) for i in list(vocab)}, ignore_index=True)
    word_occrencies_df = word_occrencies_df.transpose()
    return word_occrencies_df


word_occurencies_df = visualize_word_occurencies(vocab, tokenized_docs)


print("Word occurencies: \n", word_occurencies_df.head())

# Calculate frequencies from words

def calc_frequences(vocab, data_df):
    words_frequences = {}
    words_frequences = dict.fromkeys(list(vocab), 0)
    for word in list(vocab):
        for i in data_df.loc[[word], :].sum():
            words_frequences[word] += i
    words_frequences = {k: v for k, v in sorted(words_frequences.items(), key=lambda item: item[1], reverse=True)}
    return words_frequences

words_frequences = calc_frequences(vocab, word_occurencies_df)

# Visualize most popular words

def show_plot_most_popular_words(start, end, words_frequences):
    fig, ax = plt.subplots(figsize=(16,8))
    x = list(words_frequences.keys())[start:end]
    y = list(words_frequences.values())[start:end]
    ax.bar(range(len(x)), y)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation='vertical')
    ax.set_title('Words frequences (excluding stop words)')
    ax.set_xlabel('Word');
    ax.set_ylabel('Number of occurences');
    plt.show()

start = 0
end = 8
show_plot_most_popular_words(start, end, words_frequences)

#  Remove the most frequent words, adjectives and non-informative words

remove_words = {'user', 'method', 'approach', 'result', 'model', 'data', 'problem', 'solution',
'concept', 'task', 'work', 'process', 'step', 'goal', 'entity', 'aim', 'set', 'return', 'query', 'type','mission', 'knowledge','object', 'part', 'improvement','policy','insight',
'frame','enhancement', 'measure', 'extraction', 'increase','factor', 'event', 'retrieval', 'book', 'version', 'ability', 'view',
 'stage', 'level', 'hair', 'subset', 'issue', 'custom', 'interest', 'show', 'g', 'gan', 'na', 'match', 'lack', 'ifit', 'million',
 'art', 'occurrence', 'student', 'capability', 'case', 'finding', 'https', 'github', 'com'}


important_words = {'robotics', 'cnn', 'machinelearning', 'kmeans', 'linguistic', 'aerospace', 'aircraft', 'logistics', 'transport',
'timeseries', 'artificial', 'telematics', 'tomography', 'bert', 'gameplay', 'medicine', 'gaming', 'detector', 'forecast', 'robotic', 'android',
'genetic', 'statistic', 'cybersecurity', 'nlp', 'modelling', 'convolutional', 'textual'}

def get_most_frequent_words(upper_percent, lower_percent, words_frequences, word_occurencies_df):
    number_of_docs = len(word_occurencies_df.columns)
    most_frequent_words = []
    for word in words_frequences.keys():
      if words_frequences[word] >= number_of_docs * upper_percent / 100 or words_frequences[word] <= number_of_docs * lower_percent / 100:
          most_frequent_words.append((word, words_frequences[word]))
    return most_frequent_words

upper_percent = 50
lower_percent = 2
most_frequent_words = get_most_frequent_words(upper_percent, lower_percent, words_frequences, word_occurencies_df)


# Update a set of unnecessary words with the most frequent words

def update_remove_words_set(most_frequent_words, important_words, remove_words):
    for word in dict(most_frequent_words).keys():
        if word not in important_words:
            remove_words.add(word)
    return remove_words


remove_words_updated = update_remove_words_set(most_frequent_words, important_words, remove_words)

# Update a set of unnecessary words with adjectives

def create_remove_words_set(adjectives_list, remove_words):
    for i in adjectives_list:
      remove_words.add(i)
    return remove_words

remove_words_with_adjectives = create_remove_words_set(adjectives_list, remove_words_updated)

# Delete unnecessary words from text

def delete_unnecessary_words_from_text(tokenized_docs, remove_words):
    for i in range(len(tokenized_docs)):
        for j in remove_words:
            tokenized_docs[i] = list(filter(lambda a: a != j, tokenized_docs[i]))
    return tokenized_docs

tokenized_docs_deleted_words = delete_unnecessary_words_from_text(tokenized_docs, remove_words_with_adjectives)


# Add additional stop words (they may occur in every abstract, which means they are not representatible)

additional_stop_words = ["style", "choice", 'participant',"complicated", 'project',
'performance', 'time', 'item', 'completion', 'path',
'difficulty', 'future_work', 'purpose', 'reason', 'attention', 'transfer',
'area', 'change', 'scene', 'removal', 'output', 'address', 'state', 'article',
'control', 'source', 'limitation', 'manner', 'year', 'removal', 'augmentation', 'world',
'challenge', 'fence', 'obstruction', 'ground', 'condensate', 'ab', 'experience',
'b', 'leverage', 'fluency', 'similarity', 'planning', 'extract', "question", "context","essay",
"outcome", "educator", 'advance','failure', 'library', 'input', 'reconstruction', 'yield',
'utterance', 'okta', 'usage', 'cpc', 'possibility', 'theme',"contract", "node",
"identification", "city", "inference", "gap", "way", "internship", "field","success",
"background","importance","capture","medium","point","goal_project","risk","impact","measurement",
"code", "amount", "size", "focus", "term", "range", "property", "history", "experimen",
"people", "benefit", "technique","end","presence", "target", 'age', 'practice', 'variety',
'profile', 'edge', 'relationship', 'map', 'module', 'accuracy', 'domain', 'feedback', 'partner',
'skill', 'aspect', 'effect', 'additional', 'collection', 'action', 'study',
'activity','thing','number','function','day', 'collection',
'core', 'variation', 'company', "business",'scale', 'combination', 'access',
'demand', 'person', 'framework', 'behavior', 'gain',
'requirement', 'role', 'today', "quality", "generation", "hand", "response",
"communication", "resource","error","order","life",'researcher',
'phase', 'rule', 'pattern', 'consumption', 'form', 'cost', 'research_project', 'information', 'technology', 'home', 'research',
'technical', 'interactive', 'scenario', 'important', 'educational', 'learning', 'canada', 'effective', 'university', 'active']

def update_tokenized_docs(tokenized_docs, stop_words):
    for i in range(len(tokenized_docs)):
        for j in stop_words:
            tokenized_docs[i] = list(filter(lambda a: a != j, tokenized_docs[i]))
    for i in range(len(tokenized_docs)):
        for j in range(len(tokenized_docs[i])):
            if tokenized_docs[i][j] == 'graphql' or tokenized_docs[i][j] == 'cpc':
                tokenized_docs[i][j] = 'graph'
            if tokenized_docs[i][j] == 'ml':
                tokenized_docs[i][j] = 'machine_learning'
            if tokenized_docs[i][j] == 'train':
                tokenized_docs[i][j] = 'training'
            elif tokenized_docs[i][j] == 'nlu' or tokenized_docs[i][j] == 'nlg':
                tokenized_docs[i][j] = 'natural_language'
    return tokenized_docs

tokenized_docs_updated = update_tokenized_docs(tokenized_docs_deleted_words, additional_stop_words)


# Update vocabulary and word frequencies

updated_vocab = form_vocab(tokenized_docs_updated)

def calculate_frequencies(tokenized_docs):
    dummy_frame = {'summaries': tokenized_docs}
    dummy_df = pd.DataFrame(dummy_frame)
    for i, row in dummy_df.iterrows():
        dummy_df.at[i,'summaries'] = " ".join(dummy_df.loc[i, "summaries"])
    most_popular_words = Counter(" ".join(dummy_df["summaries"]).split()).most_common(100)
    return dict(most_popular_words)


most_popular_words = calculate_frequencies(tokenized_docs_updated)

word_occurencies_updated_df = visualize_word_occurencies(updated_vocab, tokenized_docs_updated)

start = 0
end = 18
show_plot_most_popular_words(start, end, most_popular_words)





## BERT + LDA + K-Means + t-SNE
# Idea: Concatenated both BERT and LDA vectors with a weight hyperparameter to balance the relative importance of information from each source.

# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

original_tokenized_docs = summaries_for_bert.apply(lambda x: " ".join(x))

prepared_docs_for_tokenizer = summaries_for_bert.apply(lambda x: "[CLS] " + x + " [SEP]")

bert_tokenized_docs = prepared_docs_for_tokenizer.apply(lambda x: tokenizer.tokenize(x))

model = SentenceTransformer('all-mpnet-base-v2')
bert_embeddings = model.encode(bert_tokenized_docs, show_progress_bar=True) # embeddings for summaries



### LDA
# Unlike in BERT, we feed preprocessed data in LDA. get_lda_embeddings (tokenized_docs, num_topics)
# returns a matrix whose rows are documents and columns are topics with some probability.

def get_lda_embeddings(tokenized_docs, num_topics):
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    lda_model_bert = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
    n_doc = len(corpus)
    vec_lda = np.zeros((n_doc, num_topics))
    for i in range(len(corpus)):
        for topic, prob in lda_model_bert.get_document_topics(corpus[i]):
            vec_lda[i, topic] = prob
    return vec_lda, lda_model_bert

num_topics = 30
vec_lda, lda_model_bert = get_lda_embeddings(tokenized_docs_updated, num_topics)

### Combine LDA and BERT with fixed parameter *gamma*


gamma = 15
vec_ldabert = np.c_[vec_lda * gamma, bert_embeddings]

### K-Means

num_topics = 30
kmeans = KMeans(n_clusters=num_topics)
kmeans_embed = kmeans.fit(vec_ldabert)


print(kmeans_embed.labels_)

print(kmeans_embed.cluster_centers_)

### t-SNE

tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca', perplexity=5)
tsne_ldabert = tsne_model.fit_transform(vec_ldabert)


tsne_ldabert_centers = tsne_model.fit_transform(kmeans_embed.cluster_centers_)

def get_kmeans_labels_and_kmeans_centers(num_topics, y, tsne_ldabert, tsne_ldabert_centers):
    kmeans_labels = {i: [] for i in range(0, num_topics)}
    kmeans_centers = {i: 0 for i in range(0, num_topics)}
    for i in range(len(y.labels_)):
        kmeans_labels[y.labels_[i]].append([tsne_ldabert[i][0], tsne_ldabert[i][1]])
    for i in range(num_topics):
        kmeans_centers[i] = [tsne_ldabert_centers[i][0], tsne_ldabert_centers[i][1]]
    return kmeans_labels, kmeans_centers



kmeans_labels, kmeans_centers = get_kmeans_labels_and_kmeans_centers(num_topics, kmeans_embed, tsne_ldabert, tsne_ldabert_centers)

def get_kmeans_texts_and_kmeans_title(num_topics, kmeans_embed, tokenized_docs, titles):
    kmeans_texts = {i: [] for i in range(0, num_topics)}
    kmeans_title = {i: [] for i in range(0, num_topics)}
    for i in range(len(kmeans_embed.labels_)):
        kmeans_texts[kmeans_embed.labels_[i]].append(tokenized_docs[i])
        kmeans_title[kmeans_embed.labels_[i]].append(titles[i])
    return kmeans_texts, kmeans_title


kmeans_texts, kmeans_title = get_kmeans_texts_and_kmeans_title(num_topics, kmeans_embed, tokenized_docs_updated, titles)


## Create new dataframe with topic


def get_kmeans_text_freqs(num_topics, kmeans_texts):
    kmeans_text_freqs = {i: {} for i in range(num_topics)}
    for i in kmeans_texts.keys():
        for j in range(len(kmeans_texts[i])):
            key_words = kmeans_texts[i][j]
            for c in key_words:
                if kmeans_text_freqs[i].get(c) == None:
                    kmeans_text_freqs[i][c] = 1
                else:
                    kmeans_text_freqs[i][c] += 1
    return kmeans_text_freqs


kmeans_text_freqs = get_kmeans_text_freqs(num_topics, kmeans_texts)


def get_words(ind, kmeans_text_freqs):
    s = ' '
    count = 0
    for k in kmeans_text_freqs[ind].keys():
        s += k + ", "
        count += 1
        if count == 5:
            s.rstrip(', ')
            break
    return s



# def create_new_dataframe(titles, tokenized_docs, kmeans_embed, kmeans_text_freqs):
#     new_df = pd.DataFrame(columns=['Title', 'Abstract', 'Topic'])
#     num_abstract_for_visualizing = 120
#     for i in range(num_abstract_for_visualizing):
#         new_dict = {'Title': titles[i], 'Abstract': tokenized_docs[i], 'Topic': get_words(kmeans_embed.labels_[i], kmeans_text_freqs)}
#         new_df = new_df.append(new_dict, ignore_index=True)
#     return new_df

def create_new_dataframe(titles, tokenized_docs, kmeans_embed, kmeans_text_freqs):
    data_list = []
    num_abstract_for_visualizing = 120
    for i in range(num_abstract_for_visualizing):
        new_dict = {'Title': titles[i], 'Abstract': tokenized_docs[i], 'Topic': get_words(kmeans_embed.labels_[i], kmeans_text_freqs)}
        data_list.append(new_dict)

    new_df = pd.DataFrame(data_list)
    return new_df


new_df = create_new_dataframe(titles, tokenized_docs_updated, kmeans_embed, kmeans_text_freqs)


for i in kmeans_text_freqs:
    kmeans_text_freqs[i] = {k: v for k, v in sorted(kmeans_text_freqs[i].items(), key=lambda item: item[1], reverse=True)[:5]}


colors =  ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]




def create_dataframe_topic_title_keywords(num_topics, kmeans_text_freqs, kmeans_title, titles):
    doc_topics = []
    se = set()
    for i in range(num_topics):
        topic = get_words(i, kmeans_text_freqs)
        for j in range(len(kmeans_labels[i])):
            p = {}
            if kmeans_title[i][j] in list(titles[:108]) and (kmeans_title[i][j] not in se):
                p['Topic'] = topic
                se.add(kmeans_title[i][j])
                p['Title'] = kmeans_title[i][j]
                p['Doc_keywords'] = kmeans_texts[i][j]
                doc_topics.append(p)
    df = pd.DataFrame(doc_topics)
    return df


topic_title_keywords_df = create_dataframe_topic_title_keywords(num_topics, kmeans_text_freqs, kmeans_title, titles)


# topic_title_keywords_df.to_excel('topic_title_keywords_df.xlsx')


def visualize_clusters(num_topics, kmeans_text_freqs, colors, kmeans_labels, kmeans_title, titles):
    f = open('kmeans_lda2.csv', 'w')

    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(['Title', 'x', 'y', 'Topic'])
    fig, ax = plt.subplots(figsize = (20,20))
    patches = []
    for i in range(num_topics):
        topic = get_words(i, kmeans_text_freqs)
        patch = mpatches.Patch(color=colors[i], label=topic)
        patches.append(patch)
        for j in range(len(kmeans_labels[i])):
            if kmeans_title[i][j] in list(titles):
                ax.scatter(kmeans_labels[i][j][0], kmeans_labels[i][j][1], color=colors[i])
                writer.writerow([kmeans_title[i][j], kmeans_labels[i][j][0], kmeans_labels[i][j][1], topic])
    f.close()


visualize_clusters(num_topics, kmeans_text_freqs, colors, kmeans_labels, kmeans_title, titles)

pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))



plt.show()
