import numpy as np
import argparse
from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds

from WikiDataReader import *

#####################################################################
# Part 3:
# Define your new subclasses below!
#####################################################################

class SVDWordVectors(WikiWordVectors):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7)

    def train(self, documents):
        term_document_matrix = self.vectorizer.fit_transform(documents)
        U, s, Vh = svds(term_document_matrix, k=100)
        self.V = Vh
        return np.transpose(self.V)

    def get_word_index(self, word): 
        return self.vectorizer.vocabulary_.get(word, -1)

    def get_vocabulary(self): 
        return self.vectorizer.get_feature_names_out().tolist()

    def get_word_vector(self, word):
        index = self.get_word_index(word)
        if index == -1:
            return None 
        return np.transpose(self.V)[index].reshape(1, -1)







#####################################################################
# WikiWordLookup
#####################################################################

class WikiWordLookup():

    def __init__(self, articles, word_vector_model):
        """
        initializing the WikiWordLookup class with a WikiArticles object and a WikiWordVectors object
        wiki_word_vectors is initialized to None and will be set later I assume
        """
        self.articles = articles
        self.word_vector_model = word_vector_model
        self.wiki_word_vectors = None

    def train_vector_model(self):
        """
        trains the word vector model on the abstracts of the articles in the WikiArticles object
        and normalizes the resulting vectors and stores them in the wiki_word_vectors attribute
        """
        print("Training vector model. Please wait, this may take a while...")
        self.wiki_word_vectors = self.word_vector_model.train(self.articles.get_abstracts())
        self.wiki_word_vectors = normalize(self.wiki_word_vectors)
        print("Done!")

    def compute_rankings(self, input_word):
        """
        this is where we take an input word and compute the cosine similarity between the input word vector
        and the word vectors and return the similarity scores and the ranking of the words based on similarity scores 
        """
        if self.wiki_word_vectors is None:
            raise RuntimeError("Please run train_vector_model first!")

        word_vector = normalize(self.word_vector_model.get_word_vector(input_word))
        if word_vector is None:
            return None, None

        # cosine similarity between the vector for input_word and all
        # other word vectors in the self.wiki_word_vectors matrix, and
        # saves the similarity scores to a list or numpy array called 
        # sims, indexed in the same order as self.wiki_word_vectors.
        

        if hasattr(word_vector, "toarray"):
            word_vector = word_vector.toarray()

        word_vectors = self.wiki_word_vectors
        if hasattr(word_vectors, "toarray"):
            word_vectors = word_vectors.toarray()

        sims = np.dot(word_vectors, word_vector.reshape(-1, 1))
        sims = sims.flatten()

        return sims, np.argsort(sims)[::-1][1:]

    def run_search(self, input_word, n_results=10):
        """
        similar to the run_search method in the WikiSearchEngine class, but this one is for looking up similar words to an input word
        printing the top 10 results or n results and print them out with the word and similarity score
        """
        input_word = input_word.lower()
        print(f"Top {n_results} most similar words to {input_word}:")
        sims, ranking = self.compute_rankings(input_word)
        if ranking is None:
            print(f"'{input_word}' not found in the vocabulary!")
        else:
            for rank, i in enumerate(ranking[:n_results]):
                print(f"{rank+1}. {self.word_vector_model.get_vocabulary()[i]} (similarity={sims[i]})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Directory containing the Wikipedia corpus files")
    parser.add_argument("-w", "--word", type=str, metavar="W", required=True,
                        help="The word to look up")
    parser.add_argument("-n", "--num_articles", type=int, metavar="N", default=None,
                        help="Only keep the first N articles parsed from the corpus files")
    
    args = parser.parse_args()

    articles = WikiArticles(args.data_dir, args.num_articles)

    word_vector_model = SVDWordVectors()

    word_lookup = WikiWordLookup(articles, word_vector_model)
    word_lookup.train_vector_model()
    word_lookup.run_search(args.word)

if __name__ == "__main__":
    main()
