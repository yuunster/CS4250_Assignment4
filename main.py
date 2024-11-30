from database import *
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

## Store documents in mongoDB
#################################################################################################
# Documents from question 3
documents = [
    'After the medication, headache and nausea were reported by the patient.',
    'The patient reported nausea and dizziness caused by the medication.',
    'Headache and dizziness are common effects of this medication.',
    'The medication caused a headache and nausea, but no dizziness was reported.'
]
# Insert documents into the documents_collection
for i, result in enumerate(documents):
    documents_collection.update_one(
        { '_id': i },
        { '$set': { 'content': result } },
        upsert = True
    )
#################################################################################################

## Indexing
#################################################################################################
# Instantiate the vectorizer object
# Note: Automatically lowercases and removes punctuation
vectorizer = TfidfVectorizer(
    analyzer = 'word',
    ngram_range = (1, 3)  # Include unigrams, bigrams, and trigrams
)

# Build vocabulary and transform documents into vectors
sparse_matrix = vectorizer.fit_transform(documents)

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
    sparse_vector = sparse_matrix.getrow(index)
    
    # Iterate through each term and compare to this document's vector
    for term, pos in vectorizer.vocabulary_.items():
        # If this term is non-zero in this document's vector
        if pos in sparse_vector.indices:
            # Add the term to the inverted index
            inverted_index[term]['docs'].append({
                "id": index,
                "tfidf": sparse_vector[0, pos]
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

## Querying
#################################################################################################
# Hard coded queries
queries = [
    'nausea and dizziness',
    'effects',
    'nausea was reported',
    'dizziness',
    'the medication'
]

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
                "content": "$docs.id.content",
                "docId": "$docs.id._id"
            }
        }
    ]
    results = inverted_index_collection.aggregate(pipeline)

    # Organize values to easily see matched documents
    term_document_map = {}
    for result in results:
        if result['docId'] not in term_document_map:
            term_document_map[result['docId']] = {}

        term_document_map[result['docId']]['content'] = result['content']

    # Calculate cosine similarity for each matched document
    documents_with_cos_sim = []
    for key, value in term_document_map.items():
        document_vector = sparse_matrix[key]
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