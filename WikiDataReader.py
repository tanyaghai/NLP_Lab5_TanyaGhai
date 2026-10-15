import json
import os
import sys
from abc import ABC, abstractmethod

#####################################################################
# HELPER FUNCTIONS
#####################################################################

def get_files(dirname): 
    """A generator: recursively navigates a directory, yielding each file's
    full filepath instead of just the filename."""
    for root, dirs, files in os.walk(dirname): 
        for fname in files: 
            yield os.path.join(root, fname)

#####################################################################
# WikiArticles
#####################################################################

class WikiArticles():

    def __init__(self, data_dir, max_articles=None):
        self.data_dir = data_dir
        self.max_articles = max_articles
        self.article_names = None

    def _read_wiki_file(self, fp):
        for line in fp:
            line_as_json = None # TODO replace me: use the json library to parse the current line as a JSON object
            name = None         # TODO replace me: in the JSON object you got above, look up the value of the "name" attribute
            abstract = None     # TODO replace me: in the JSON object you got above, look up the value of the "abstract" attribute
            yield name, abstract

    def get_abstracts(self):
        n_processed = 0
        for fname in get_files(self.data_dir):
            with open(fname) as fp:
                for name, abstract in self._read_wiki_file(fp):

                    if self.max_articles is not None and n_processed > self.max_articles:
                        break
                    n_processed += 1

                    if self.article_names is None:
                        self.article_names = []

                    self.article_names.append(name)
                    yield abstract

    def get_names(self):
        if self.article_names is None:
            raise RuntimeError("get_names cannot be called until get_abstracts has been called!")
        return self.article_names

#####################################################################
# WikiDocumentVectors
#####################################################################

class WikiDocumentVectors(ABC):

    @abstractmethod
    def train(self, documents):
        """
        Trains a vector semantics model on the input documents,
        and returns a matrix containing the resulting vector
        representations of each document. The matrix should be
        indexed the same as the input documents (i.e., the ith row
        should be the vector representation of the ith document)

        Parameters:
        documents: an iterable over documents, where each document is represented as a string

        Returns:
        A matrix containing the vector representations of each document, such that the ith row is the vector for the ith document
        """
        raise NotImplementedError()

    @abstractmethod
    def get_document_vector(self, new_doc):
        """
        Using the trained vector semantics model, computes and returns
        the vector representation of a new input document (i.e., a
        document that was not previously seen during training). This
        method can only be called after train has been called.

        Parameters:
        new_doc: a string representing the new document to process

        Returns:
        The vector representation of new_doc, as computed by the vector semantics model
        """
        raise NotImplementedError()

#####################################################################
# WikiWordVectors
#####################################################################

class WikiWordVectors(ABC):

    @abstractmethod
    def train(self, documents):
        """
        Trains a vector semantics model on the input documents,
        and returns a tuple containing (1) a vocabulary learned
        from the input documents, and (2) a matrix containing the
        resulting vector representations of each word in the vocabulary
        (i.e., the ith row of the matrix should be the vector representation
        of the ith vocabulary word)

        Parameters:
        documents: an iterable over documents, where each document is represented as a string

        Returns:
        A tuple containing:
        - A vocabulary learned from the input documents
        - A matrix where the ith row is the vector representation of the ith vocab word
        """
        raise NotImplementedError()

    @abstractmethod
    def get_word_index(self, word):
        """
        Returns:
        The index of the word in the vocabulary that was learned during train, or -1 if the word isn't in the vocab
        """
        raise NotImplementedError()

    @abstractmethod
    def get_vocabulary(self):
        """
        Returns:
        The vocabulary that was learned during train, as a list of words
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_word_vector(self, word):
        """
        Given a word that is in the vocabulary learned during train,
        looks up and returns the vector for that word. This
        method can only be called after train has been called.

        Parameters:
        word: a string representing the word that we want the vector for

        Returns:
        The vector representation of word, or None if word is not in the vocab.
        """
        raise NotImplementedError()
