import numpy as np
import argparse
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds

from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer

from WikiDataReader import *

#####################################################################
# Part 2:
# Define your new subclasses below!
#####################################################################

class CountDocumentVectors(WikiDocumentVectors):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7)

    def train(self, documents):
        return self.vectorizer.fit_transform(documents)

    def get_document_vector(self, document):
        return self.vectorizer.transform([document])


class SVDDocumentVectors(WikiDocumentVectors):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7)

    def train(self, documents):
        term_document_matrix = self.vectorizer.fit_transform(documents)
        U, s, Vh = svds(term_document_matrix, k=100)
        self.V = Vh
        return U


    def get_document_vector(self, document):
        query_vector = self.vectorizer.transform([document])
        return query_vector @ np.transpose(self.V)

class NeuralDocumentVectors(WikiDocumentVectors):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def train(self, documents):
        return self.model.encode(
            list(documents),
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def get_document_vector(self, document):
        return self.model.encode([document], convert_to_numpy=True)



#####################################################################
# WikiSearchEngine
#####################################################################

class WikiSearchEngine():

    def __init__(self, articles, doc_vector_model):
        """
        initializes the WikiSearchEngine with a WikiArticles object and a WikiDocumentVectors object
        wiki_corpus_vectors is initialized to None and will be set later I assume

        """
        self.articles = articles
        self.doc_vector_model = doc_vector_model
        self.wiki_corpus_vectors = None

    def train_vector_model(self):
        """
        trains the document vector model on the abstracts of the articles in the WikiArticles object
        and normalizes the resulting vectors and stores them in the wiki_corpus_vectors attribute

        so this gets a matrix of document vectors for the entire corpus of articles and each row 
        is the vector representation of the corresponding article in the WikiArticles object
        the columns of the matrix are the words in the vocabulary of the document vector model


        """
        print("Training vector model. Please wait, this may take a while...")
        self.wiki_corpus_vectors = self.doc_vector_model.train(self.articles.get_abstracts())
        self.wiki_corpus_vectors = normalize(self.wiki_corpus_vectors)
        print("Done!")

    def compute_rankings(self, query):

        """
        this is where we take a query and compute the cosine similarity between the query vector 
        and the document vectors and return the similarity scores and the ranking of the documents based on similarty scores 

        """
        if self.wiki_corpus_vectors is None:
            raise RuntimeError("Please run train_vector_model first!")

        query_vector = normalize(self.doc_vector_model.get_document_vector(query))

        # computes the
        # cosine similarity between the query vector and all document
        # vectors in the self.wiki_corpus_vectors matrix, and saves
        # the similarity scores to a list or numpy array called sims,
        # indexed in the same order as self.wiki_corpus_vectors.

    
        if hasattr(query_vector, "toarray"):
            query_vector = query_vector.toarray()

        document_vectors = self.wiki_corpus_vectors
        if hasattr(document_vectors, "toarray"):
            document_vectors = document_vectors.toarray()

        sims = np.dot(document_vectors, query_vector.reshape(-1, 1))
        sims = sims.flatten() 

        

        return sims, np.argsort(sims)[::-1]

    def run_search(self, query, n_results=10):
        """
        this is where we take a query and compute the cosine similarity between the query vector
        we print the top 10 results or n results and print them out with the article name and similarity score 
        """
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

    parser.add_argument("-m", "--model", type=str,
                    choices=["tfidf", "svd", "neural"],
                    default="tfidf",
                    help="Document vector model to use")              
    
    args = parser.parse_args()

    articles = WikiArticles(args.data_dir, args.num_articles)

    if args.model == "svd":
        doc_vector_model = SVDDocumentVectors()
    elif args.model == "neural":
        doc_vector_model = NeuralDocumentVectors()
    else:
        doc_vector_model = CountDocumentVectors()

    my_google = WikiSearchEngine(articles, doc_vector_model)
    my_google.train_vector_model()
    my_google.run_search(args.query)

if __name__ == "__main__":
    main()
