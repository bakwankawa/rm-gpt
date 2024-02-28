import os, dotenv
from typing import Optional
from langchain.vectorstores import FAISS

from langchain.embeddings.openai import OpenAIEmbeddings
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

import csv

from langchain.evaluation import load_evaluator
from langchain.evaluation import EmbeddingDistance

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings()

import csv
import os

# Evaluation variable
evaluation_data = []

# Load the evaluator
# evaluator = load_evaluator("embedding_distance", distance_metric=EmbeddingDistance.EUCLIDEAN)
# evaluator = load_evaluator("embedding_distance", distance_metric=EmbeddingDistance.COSINE)
evaluator = load_evaluator("embedding_distance", distance_metric=EmbeddingDistance.MANHATTAN)

# Initialize OpenAIEmbeddings and local_vector_database
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# pickle_path = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\fafa faiss pkl\faiss_index"
pickle_path = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\RM-GPT\faiss_index"
local_vector_database = FAISS.load_local(pickle_path, embeddings)
top_k = 10

# Function to save evaluation data to CSV
def save_to_csv(filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["User Input", "Chatbot Response", "Retrieval", "Best Similarity Score", "Precise?"])

        for question, chatbot_response_str, best_retrieval, best_score, precision in evaluation_data:
            csv_writer.writerow([question, chatbot_response_str, best_retrieval, best_score, precision])

def run_retrieval(question):
    docs = local_vector_database.similarity_search(question, k=top_k)
    # top_score = docs[0].page_content
    return docs

if __name__ == "__main__":
    try:
        csv_file_path = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\RM-GPT\RM-GPT\evaluation\RM-GPT test - specific product.csv"  # Replace with the actual path
        questions = []
        chatbot_responses = []

        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                questions.append(row['question'])
                chatbot_responses.append(row['chatbot_response'])  # Store chatbot responses

        for question, chatbot_response in zip(questions, chatbot_responses):
            retrieval_docs = run_retrieval(question)

            # Convert chatbot_response to string
            chatbot_response_str = str(chatbot_response)

            # Initialize variables
            best_score = float('inf')  # Initialize with a high value
            best_retrieval = None

            # Compare chatbot_response with each retrieval_doc
            for doc in retrieval_docs:
                retrieval_doc_str = str(doc.page_content)  # Convert the whole document to string
                score_dict = evaluator.evaluate_strings(prediction=chatbot_response_str, reference=retrieval_doc_str)
                score = score_dict.get('score', float('inf'))
                print("\n score: ", score)

                # Update if the score is better
                if score < best_score:
                    best_score = score
                    best_retrieval = retrieval_doc_str

            # Determine if best_score > 10
            if best_score < 12:
                precision = "Yes"
            else:
                precision = "No"
                    
            print("\n best retrieval: ", best_retrieval)
            print("\n best score: ", best_score)

            # Append data to evaluation_data
            evaluation_data.append((question, chatbot_response_str, best_retrieval, best_score, precision))  # Add data for evaluation

    except KeyboardInterrupt:
        print("\nChatbot terminated by user.")

    # Save evaluation data to CSV when the loop is done
    save_directory = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\RM-GPT\RM-GPT\evaluation"
    csv_filename = os.path.join(save_directory, "specific_product.csv")
    save_to_csv(csv_filename)
