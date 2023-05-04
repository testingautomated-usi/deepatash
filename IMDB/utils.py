import random
import matplotlib
matplotlib.use('Agg')
import logging as log
import sys

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

#from tensorflow import keras
from config import DIVERSITY_METRIC
import numpy as np
import re
import Levenshtein as lev

def edit_distance(txt1, txt2):
    preprocess_text1 = text_prep(txt1)
    pos_words1 = len([i for i in preprocess_text1 if i in pos_words])
    neg_words1 = len([i for i in preprocess_text1 if i in neg_words])
    sentiment_words1 = abs(pos_words1 - neg_words1)

    preprocess_text2 = text_prep(txt2)
    pos_words2 = len([i for i in preprocess_text2 if i in pos_words])
    neg_words2 = len([i for i in preprocess_text2 if i in neg_words])
    sentiment_words2 = abs(pos_words2 - neg_words2)

    distance = abs(sentiment_words2 - sentiment_words1)

    return distance


def get_element_by_seed(fm, seed):
    for (x,y), value in np.ndenumerate(fm):
        if value != None:
            for v in value:
                if v.seed == seed:
                    return (x,y)
    return None

def get_distance(ind1, ind2):
    """ Computes distance based on configuration """


    if DIVERSITY_METRIC == "INPUT":
        # input space
        distance = lev.distance(ind1.text, ind2.text)
    
    elif DIVERSITY_METRIC == "LATENT":
        # latent space
        distance = euclidean(ind1.latent_vector, ind2.latent_vector)


    elif DIVERSITY_METRIC == "HEATMAP":
        # heatmap space
        distance = euclidean(ind1.explanation, ind2.explanation)


    elif DIVERSITY_METRIC == "HEATLAT":
        # latent space
        distance = euclidean(ind1.heatmap_latent_vector, ind2.heatmap_latent_vector)
    

    return distance


def get_distance_by_metric(ind1, ind2, metric):
    """ Computes distance based on metric """

    if metric == "INPUT":
        # input space
        distance = lev.distance(ind1.text, ind2.text)
    
    elif metric == "LATENT":
        # latent space
        distance = euclidean(ind1.latent_vector, ind2.latent_vector)


    elif metric == "HEATMAP":
        # heatmap space
        distance = euclidean(ind1.explanation, ind2.explanation)

    elif metric == "HEATLAT":
        # latent space
        distance = euclidean(ind1.heatmap_latent_vector, ind2.heatmap_latent_vector)
    

    return distance


def kl_divergence(ind1, ind2):
    mu1 = ind1[0]
    sigma_1 = ind1[1]
    mu2 = ind2[0]
    sigma_2 = ind2[1]

    sigma_diag_1 = np.eye(sigma_1.shape[0]) * sigma_1
    sigma_diag_2 = np.eye(sigma_2.shape[0]) * sigma_2
    sigma_diag_2_inv = np.linalg.inv(sigma_diag_2)

    kl = 0.5 * (np.log(np.linalg.det(sigma_diag_2) / np.linalg.det(sigma_diag_2)) 
        - mu1.shape[0] + np.trace(np.matmul(sigma_diag_2_inv, sigma_diag_1))  
        + np.matmul(np.matmul(np.transpose(mu2 - mu1),sigma_diag_2_inv), (mu2 - mu1)))
    return kl


def euclidean(img1, img2):
    dist = np.linalg.norm(img1 - img2)
    return dist

def manhattan(coords_ind1, coords_ind2):
    return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1])





from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')


file = open('opinion-lexicon-English/negative-words.txt', 'r')
neg_words = file.read().split()

file = open('opinion-lexicon-English/positive-words.txt', 'r')
pos_words = file.read().split()


def untokenize(vector):
    return TreebankWordDetokenizer().detokenize(vector)    


def get_synonym(word):
    word = word.lower()
    synonyms = []
    synsets = wordnet.synsets(word)
    if (len(synsets) == 0):
        return []
    for synset in synsets:
        lemma_names = synset.lemma_names()
        for lemma_name in lemma_names:
            lemma_name = lemma_name.lower().replace('_', ' ')
            if (lemma_name != word and lemma_name not in synonyms):
                synonyms.append(lemma_name)
    if len(synonyms) > 0:
        sword = random.choice(synonyms)
        return sword
    else:
        return None


def listToString(s): 
    
    # initialize an empty string
    str1 = s[0] 
    
    # traverse in the string  
    for ele in s[1:]: 
        if isinstance(ele, str):
            str1 += "." + ele  
    
    # return string  
    return str1 

def find_adjs(text):
    tokenized_text = word_tokenize(text)
    word_tags = nltk.pos_tag(tokenized_text)
    adjs_advs = [i for i in range(0,len(word_tags)) if word_tags[i][1] in ['JJ', 'JJR', 'JJS']]
    return word_tags, adjs_advs

def text_prep(x):
    corp = str(x).lower()
    corp = re.sub('[^a-zA-Z]+', ' ', corp).strip()
    tokens = word_tokenize(corp)
    words = [t for t in tokens if t not in stop_words]
    lemmatize = [lemma.lemmatize(w) for w in words]
    return lemmatize

def count_pos(text):
    
    preprocess_text = text_prep(text)
    num_pos = len([i for i in preprocess_text if i in pos_words])
    return num_pos

def count_neg(text):
    preprocess_text = text_prep(text)
    num_neg = len([i for i in preprocess_text if i in neg_words])
    return num_neg


def count_verbs(text):
    tokenized_text = word_tokenize(text)
    word_tags = nltk.pos_tag(tokenized_text)
    verbs = [wt[0] for wt in word_tags if wt[1] in ['VB', 'VBD', 'VBN', 'VBP', 'VBZ', 'VBG']]
    return len(verbs)


def feature_simulator(function, x):
    """
    Calculates the value of the desired feature
    :param function: name of the method to compute the feature value
    :param x: genotype of candidate solution x
    :return: feature value
    """
    if function == 'count_neg':
        return count_neg(x.text)
    if function == 'count_pos':
        return count_pos(x.text)
    if function == 'count_verbs':
        return count_verbs(x.text)



def setup_logging(log_to, debug):

    def log_exception(extype, value, trace):
        log.exception('Uncaught exception:', exc_info=(extype, value, trace))

    # Disable annoyng messages from matplot lib.
    # See: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
    log.getLogger('matplotlib.font_manager').disabled = True

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, 'a', 'utf-8')
        log_handlers.append( file_handler )
        start_msg += " ".join(["writing to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=log_level, handlers=log_handlers)

    sys.excepthook = log_exception

    log.info(start_msg)