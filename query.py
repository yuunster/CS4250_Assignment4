from typing import cast
from database import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle
import numpy as np

## Querying
####################################################################################

# Initialize vectorizer using the stored vectorizer in mongoDB
vectorizer_document = vectorizer_collection.find_one()
vectorizer = pickle.loads(vectorizer_document['data'])
vectorizer: TfidfVectorizer = cast(TfidfVectorizer, vectorizer)

def query(queries: list[str]):
    for i, query in enumerate(queries):
        # Calculate vector for each query
        query_vector = vectorizer.transform([query])

        # Find matching documents for query
        # Extract non-zero indices
        non_zero_indices = query_vector.nonzero()[1].tolist()

        pipeline = [
            # Match documents in inverted_index_collection based on query's non zero indices
            {
                "$match": {
                    "pos": { "$in": non_zero_indices }
                }
            },
            # Unwind the 'docs' array to make each document a separate entry
            {
                "$unwind": "$docs"
            },
            # Reference documents collection to get content of each doc
            {
                "$lookup": {
                    "from": "documents",  # The collection to join with
                    "localField": "docs.id",  # Field in inverted_index to match
                    "foreignField": "_id",  # Field in documents to match
                    "as": "docs.id"  # The name of the new array field to store matched documents
                }
            },
            {
                "$unwind": "$docs.id"
            },
            # Restructure results
            {
                "$project": {
                    "_id": 0,
                    "pos": 1,
                    "tfidf": "$docs.tfidf",
                    "content": "$docs.id.content",
                    "docId": "$docs.id._id"
                }
            }
        ]
        results = inverted_index_collection.aggregate(pipeline)

        # Organize values for easier sparse vector construction
        term_document_map = {}
        for result in results:
            if result['docId'] not in term_document_map:
                term_document_map[result['docId']] = { 'pos_values': [], 'tfidf_values': [] }

            term_document_map[result['docId']]['pos_values'].append(result['pos'])
            term_document_map[result['docId']]['tfidf_values'].append(result['tfidf'])
            term_document_map[result['docId']]['content'] = result['content']

        # Create a document vector for each matched document
        documents_with_cos_sim = []
        for value in term_document_map.values():
            document_vector = csr_matrix((value['tfidf_values'], value['pos_values'], [0, len(value['tfidf_values'])]), shape=(1, query_vector.shape[1]))
            cos_sim = cosine_similarity(query_vector, document_vector)
            documents_with_cos_sim.append((
                value['content'],
                float(cos_sim[0][0])
            ))
        
        # Sort documents by cosine similarity score
        ranked_documents = sorted(documents_with_cos_sim, key=lambda x: x[1], reverse=True)

        print(f'query q{i + 1}:')
        for tuple in ranked_documents:
            print(f'"{tuple[0]}", {tuple[1]:.2f}')

        print()