from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from database import *

## Indexing
####################################################################################
def index(documents: list[str]):
    # Instantiate the vectorizer object
    # Note: Automatically lowercases and removes punctuation
    vectorizer = TfidfVectorizer(
        analyzer = 'word',
        ngram_range = (1, 3)  # Include unigrams, bigrams, and trigrams
    )

    # Build vocabulary and transform documents into vectors
    sparse_matrix = vectorizer.fit_transform(documents)

    # Serialize and store the vectorizer in mongoDB to later reinitialize a new TfidVectorizer for querying
    vectorizer_data = pickle.dumps(vectorizer)
    vectorizer_collection.update_one(
        { '_id': 'vectorizer' },  # manually set _id so that we can overwrite this doc
        { '$set': { 'data': vectorizer_data } },
        upsert = True
    )

    # Retrieve the terms after tokenization
    terms = vectorizer.get_feature_names_out()

    print(terms)

    # Print term matrix using a dataframe for console print formatting
    print("TD-IDF Vectorizer Training\n")
    print(pd.DataFrame(data = sparse_matrix.toarray(), columns = terms))

    # Initialize inverted_index with _id values
    vocabulary = vectorizer.vocabulary_
    inverted_index = {}
    for index, (term, pos) in enumerate(vocabulary.items()):
        inverted_index[term] = { '_id': index, 'pos': pos, 'docs': [] }

    # Iterate through every token
    for index, text in enumerate(documents):
        sparse_vector = sparse_matrix[index]
        
        # Iterate through each term and compare to this document's vector
        for term, pos in vectorizer.vocabulary_.items():
            # If this term is non-zero in this document's vector
            if pos in sparse_vector.indices:
                # Extract the tfidf value for that term
                tfidf = sparse_vector.data[sparse_vector.indices == pos][0]

                # Add the term to the inverted index
                inverted_index[term]['docs'].append({
                    "id": index,
                    "tfidf": tfidf
                })

    # Print terms and list of associated docs
    for term, fields in inverted_index.items():
        print(term + ": ")
        for doc in fields['docs']:
            print(doc)
        print()

    # Store the inverted_index in mongoDB
    for term_fields in inverted_index.values():
        inverted_index_collection.update_one(
            { '_id': term_fields['_id'] },
            { '$set': term_fields },
            upsert = True
        )