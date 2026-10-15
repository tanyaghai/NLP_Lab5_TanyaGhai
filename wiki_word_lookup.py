import numpy as np
import argparse
from sklearn.preprocessing import normalize

from WikiDataReader import *

#####################################################################
# Part 3:
# Define your new subclasses below!
#####################################################################






#####################################################################
# WikiWordLookup
#####################################################################

class WikiWordLookup():

    def __init__(self, articles, word_vector_model):
        self.articles = articles
        self.word_vector_model = word_vector_model
        self.wiki_word_vectors = None

    def train_vector_model(self):
        print("Training vector model. Please wait, this may take a while...")
        self.wiki_word_vectors = self.word_vector_model.train(self.articles.get_abstracts())
        self.wiki_word_vectors = normalize(self.wiki_word_vectors)
        print("Done!")

    def compute_rankings(self, input_word):
        if self.wiki_word_vectors is None:
            raise RuntimeError("Please run train_vector_model first!")

        word_vector = normalize(self.word_vector_model.get_word_vector(input_word))
        if word_vector is None:
            return None, None

        # TODO: REPLACE the following line with code that computes the
        # cosine similarity between the vector for input_word and all
        # other word vectors in the self.wiki_word_vectors matrix, and
        # saves the similarity scores to a list or numpy array called 
        # sims, indexed in the same order as self.wiki_word_vectors.
        sims = []

        return sims, np.argsort(sims)[::-1][1:]

    def run_search(self, input_word, n_results=10):
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
    # TODO replace the following line with an initialization of the appropriate WikiWordVectors subclass
    word_vector_model = None

    word_lookup = WikiWordLookup(articles, word_vector_model)
    word_lookup.train_vector_model()
    word_lookup.run_search(args.word)

if __name__ == "__main__":
    main()
