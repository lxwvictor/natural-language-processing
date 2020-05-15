import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import datapath


# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    #'WORD_EMBEDDINGS': 'word_embeddings.tsv',
    'WORD_EMBEDDINGS': 'GoogleNews-vectors-negative300.bin',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    ########################

    #embeddings = KeyedVectors.load_word2vec_format(datapath(embeddings_path), limit=500000, binary=True)
    embeddings = KeyedVectors.load_word2vec_format(embeddings_path, limit=500000, binary=True)
    embeddings_dim = 300

    return embeddings, embeddings_dim
    # remove this when you're done
    #raise NotImplementedError(
    #    "Open utils.py and fill with your code. In case of Google Colab, download"
    #    "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
    #    "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    # Hint: you have already implemented exactly this function in the 3rd assignment.

    ########################
    #### YOUR CODE HERE ####
    ########################
    question_vector = np.zeros(300)

    words = question.split(" ")
    valid_words = 0


    for word in words:
      try :
        word_vector = embeddings.get_vector(word.strip())
        question_vector = question_vector + word_vector
        valid_words = valid_words + 1
      except KeyError:
        None
        #print('word not found')


    if valid_words > 0:
      question_vector = question_vector / valid_words

    #print(question)
    #print(question_vector)

    return question_vector

    # remove this when you're done
    #raise NotImplementedError(
    #    "Open utils.py and fill with your code. In case of Google Colab, download"
    #    "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
    #    "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)