import numpy as np
import argparse
from sklearn.preprocessing import normalize

from WikiDataReader import *

#####################################################################
# Part 2:
# Define your new subclasses below!
#####################################################################






#####################################################################
# WikiSearchEngine
#####################################################################

class WikiSearchEngine():

    def __init__(self, articles, doc_vector_model):
        self.articles = articles
        self.doc_vector_model = doc_vector_model
        self.wiki_corpus_vectors = None

    def train_vector_model(self):
        print("Training vector model. Please wait, this may take a while...")
        self.wiki_corpus_vectors = self.doc_vector_model.train(self.articles.get_abstracts())
        self.wiki_corpus_vectors = normalize(self.wiki_corpus_vectors)
        print("Done!")

    def compute_rankings(self, query):
        if self.wiki_corpus_vectors is None:
            raise RuntimeError("Please run train_vector_model first!")

        query_vector = normalize(self.doc_vector_model.get_document_vector(query))

        # TODO: REPLACE the following line with code that computes the
        # cosine similarity between the query vector and all document
        # vectors in the self.wiki_corpus_vectors matrix, and saves
        # the similarity scores to a list or numpy array called sims,
        # indexed in the same order as self.wiki_corpus_vectors.
        sims = []

        return sims, np.argsort(sims)[::-1]

    def run_search(self, query, n_results=10):
        print(f"Top {n_results} search results for query {query}:")
        sims, ranking = self.compute_rankings(query)
        for rank, i in enumerate(ranking[:n_results]):
            print(f"{rank+1}. {self.articles.get_names()[i]} (similarity={sims[i]})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Directory containing the Wikipedia corpus files")
    parser.add_argument("-q", "--query", type=str, metavar="Q", required=True,
                        help="The search query to run")
    parser.add_argument("-n", "--num_articles", type=int, metavar="N", default=None,
                        help="Only keep the first N articles parsed from the corpus files")
    
    args = parser.parse_args()

    articles = WikiArticles(args.data_dir, args.num_articles)
    # TODO replace the following line with an initialization of the appropriate WikiDocumentVectors subclass
    doc_vector_model = None

    my_google = WikiSearchEngine(articles, doc_vector_model)
    my_google.train_vector_model()
    my_google.run_search(args.query)

if __name__ == "__main__":
    main()
