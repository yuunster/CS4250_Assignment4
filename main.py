from database import *
from indexer import index
from query import query

## Store documents in mongoDB
####################################################################################
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
####################################################################################

## Indexing
index(documents)

## Querying
# Hard coded queries
queries = [
    'nausea and dizziness',
    'effects',
    'nausea was reported',
    'dizziness',
    'the medication'
]

query(queries)