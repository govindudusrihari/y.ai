# y.ai
import firebase_admin
from firebase_admin import credentials, firestore
from pinecone import Client, Vector
from PyPDF2 import PdfReader

# Initialize Firebase
cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Initialize Pinecone
client = Client(api_key="your_pinecone_api_key", environment="us-central1-gcp")

def upload_and_index_document(document_data, chat_name):
    # Extract text from the PDF
    with open(document_data, 'rb') as pdf_file:
        reader = PdfReader(pdf_file)
        text = ''.join(page.extract_text() for page in reader.pages)

    # Create a vector representation using a suitable embedding model
    # (e.g., from Google's Gemini API)
    embedding_vector = create_embedding_vector(text)

    # Index in Pinecone
    index_name = "your_pinecone_index_name"
    vector = Vector(values=embedding_vector)
    client.upsert(index_name, [vector], [document_data])

    # Store metadata in Firebase
    db = firestore.client()
    doc_ref = db.collection("documents").document(document_data)
    doc_ref.set({
        "chat_name": chat_name,
        "pinecone_id": document_data  # Assuming document_data is a unique identifier
    })

# Example usage:
document_data = "path/to/your/pdf_document.pdf"
chat_name = "my_chat"
upload_and_index_document(document_data, chat_name
import firebase_admin
from firebase_admin import credentials, firestore
from pinecone import Client
from google.cloud import texttospeech

# Initialize Firebase
cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Initialize Pinecone
client = Client(api_key="your_pinecone_api_key", environment="us-central1-gcp")

def query_documents(chat_name, question):
    # Retrieve document index from Firebase
    db = firestore.client()
    doc_ref = db.collection("documents").where("chat_name", "==", chat_name).get()
    if len(doc_ref) == 0:
        return "Document not found."

    document_data = doc_ref[0].to_dict()["pinecone_id"]

    # Query Pinecone
    index_name = "your_pinecone_index_name"
    query_vector = create_embedding_vector(question)  # Assuming you have this function
    result = client.query(index_name, query_vector, top_k=10)

    # Retrieve relevant sections
    relevant_sections = []
    for match in result.matches:
        # Extract relevant sections based on the match (e.g., using proximity or other criteria)
        relevant_sections.append(extract_relevant_section(match.id))

    # Use Google Gemini to generate a response
    response = generate_response_with_gemini(relevant_sections, question)

    return response

# Example usage:
chat_name = "my_chat"
question = "What is the capital of France?"
answer = query_documents(chat_name, question)
print(answer
import firebase_admin
from firebase_admin import credentials, firestore
from pinecone import Client
from google.cloud import texttospeech
import re

# Initialize Firebase
cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Initialize Pinecone
client = Client(api_key="your_pinecone_api_key", environment="us-central1-gcp")

def query_documents(chat_name, question):
    # Validate question
    if not validate_question(question):
        return "Invalid question. Please rephrase your query or provide more context."

    # Retrieve document index from Firebase
    db = firestore.client()
    doc_ref = db.collection("documents").where("chat_name", "==", chat_name).get()
    if len(doc_ref) == 0:
        return "Document not found."

    document_data = doc_ref[0].to_dict()["pinecone_id"]

    # Query Pinecone
    index_name = "your_pinecone_index_name"
    query_vector = create_embedding_vector(question)  # Assuming you have this function
    result = client.query(index_name, query_vector, top_k=10)

    # Retrieve relevant sections
    relevant_sections = []
    for match in result.matches:
        # Extract relevant sections based on the match (e.g., using proximity or other criteria)
        relevant_sections.append(extract_relevant_section(match.id))

    # Use Google Gemini to generate a response
    response = generate_response_with_gemini(relevant_sections, question)

    return response

def validate_question(question):
    # Check for empty strings and excessive length
    if not question or len(question) < 5:
        return False

    # Check for profanity or offensive language (consider using a profanity filter library)
    if contains_profanity(question):
        return False

    # Check for relevance to the document content (e.g., using semantic similarity)
    # ... (implement your relevance check logic here)

    return True

def contains_profanity(text):
    # Use a profanity filter library or define your own list of prohibited words
    profanity_words = ["profanity1", "profanity2", ...]
    for word in profanity_words:
        if word in text.lower():
            return True
    return False

# Example usage:
chat_name = "my_chat"
question = "What is the capital of France?"
answer = query_documents(chat_name, question)
print(answer)
