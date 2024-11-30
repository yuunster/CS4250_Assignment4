from pymongo import MongoClient

# Create a database connection object using pymongo
DB_NAME = "cs4250_assignment4"
DB_HOST = "localhost"
DB_PORT = 27017
try:
    client = MongoClient(host=DB_HOST, port=DB_PORT)
    db = client[DB_NAME]
    documents_collection = db['documents']
    inverted_index_collection = db['invertedIndex']
except:
    print("Database not connected successfully")