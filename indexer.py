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

    # Store the document vectors in the documents collection for each doc
    for index in range(sparse_matrix.shape[0]):
        sparse_vector = sparse_matrix.getrow(index)

        # convert sparse_vector into a dictionary so we can directly insert into a mongodb document
        indices = [str(term_index) for term_index in sparse_vector.indices]   # converts np.int32 indices into strings to be used as mongoDB fields
        values = sparse_vector.data
        sparse_vector_to_dict = dict(zip(indices, values))

        # store sparse vector and doc id in mongodb documents collection
        documents_collection.update_one(
            { '_id': index },
            { '$set': { 'vector': sparse_vector_to_dict } },
            upsert = True
        )


    # Serialize and store the vectorizer in mongoDB to later reinitialize a new TfidVectorizer for querying
    vectorizer_data = pickle.dumps(vectorizer)
    vectorizer_collection.update_one(
        { '_id': 'vectorizer' },  # manually set _id so that we can overwrite this doc
        { '$set': { 'data': vectorizer_data } },
        upsert = True
    )

    # Retrieve the terms after tokenization
    terms = vectorizer.get_feature_names_out()

    # Print term matrix using a dataframe for console print formatting
    # print("TD-IDF Vectorizer Training\n")
    # print(pd.DataFrame(data = sparse_matrix.toarray(), columns = terms))

    # Initialize inverted_index with _id values
    vocabulary = vectorizer.vocabulary_
    inverted_index = {}
    for index, (term, pos) in enumerate(vocabulary.items()):
        inverted_index[term] = { '_id': index, 'pos': pos, 'docs': [] }

    # Iterate through every token
    for index, _ in enumerate(documents):
        sparse_vector = sparse_matrix[index]
        
        # Iterate through each term and compare to this document's vector
        for term, pos in vectorizer.vocabulary_.items():
            # If this term is non-zero in this document's vector
            if pos in sparse_vector.indices:
                # Add the term to the inverted index
                inverted_index[term]['docs'].append({
                    "id": index
                })

    # Print terms and list of associated docs
    # for term, fields in inverted_index.items():
    #     print(term + ": ")
    #     for doc in fields['docs']:
    #         print(doc)
    #     print()

    # Store the inverted_index in mongoDB
    for term_fields in inverted_index.values():
        inverted_index_collection.update_one(
            { '_id': term_fields['_id'] },
            { '$set': term_fields },
            upsert = True
        )