import os, dotenv
from typing import Optional
from langchain.vectorstores import FAISS

from langchain.embeddings.openai import OpenAIEmbeddings
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

import csv

from langchain.evaluation import load_evaluator
from langchain.evaluation import EmbeddingDistance

dotenv.load_dotenv("OPENAI_API_KEY")

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
# pickle_path = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\fafa faiss pkl\13082023_knowledgeall"
# pickle_path = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\fafa faiss pkl\14082023_knowledgeall"
pickle_path = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\fafa faiss pkl\15082023_knowledgeall"
# pickle_path = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\RM-GPT\faiss_index"
local_vector_database = FAISS.load_local(pickle_path, embeddings)
top_k = 10

# Function to save evaluation data to CSV
def save_to_csv(filename, header_row, data_rows):
    with open(filename, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header_row)

        for row in data_rows:
            csv_writer.writerow(row)

def run_retrieval(question):
    docs = local_vector_database.similarity_search(question, k=top_k)
    # top_score = docs[0].page_content
    return docs

if __name__ == "__main__":
    try:
        csv_file_path = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\RM-GPT\RM-GPT\evaluation\synthetic 100\testingpagi.csv"  # Replace with the actual path
        questions = []

        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                questions.append(row['question'])

        # Ambil dokumen retrievel sesuai jumlah dokumen retrieval_docs
        num_retrievals = top_k
        # retrievals = [f"retrieval {i}" for i in range(num_retrievals, 0, -1)]
        retrievals = [f"retrieval {i}" for i in range(1, num_retrievals + 1)]
        retrieval_accs = [f"retrieval acc {i}" for i in range(num_retrievals, 0, -1)]  # New column names

        for question in questions:
            retrieval_docs = run_retrieval(question)

            # Simpan data retrievel
            retrieval_data = []

            # Ambil data retrievel
            for doc in retrieval_docs:
                retrieval_data.append(str(doc.page_content))  # Konversi dokumen retrievel menjadi string
                print(doc.page_content)

            # Tambahkan data ke evaluation_data
            evaluation_data.append([question, *retrieval_data])

    except KeyboardInterrupt:
        print("\nChatbot terminated by user.")

    # Save evaluation data to CSV when the loop is done
    save_directory = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\RM-GPT\RM-GPT\evaluation\synthetic 100"
    csv_filename = os.path.join(save_directory, "added_synthetic_result.csv")

    # Create header row
    header_row = ["User Input"] + retrievals + retrieval_accs

    # Call save_to_csv with header row and data rows
    save_to_csv(csv_filename, header_row, evaluation_data)
