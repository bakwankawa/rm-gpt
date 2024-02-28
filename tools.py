from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
# import zipfile

def get_retrieval():
#   # Replace 'file_name.zip' with the actual name of your ZIP file
#   zip_file_name = '/content/db.zip'
#   # zip_file_name = '/content/faiss_index.zip'

  # Replace '/path/to/extract/folder' with the desired folder path where you want to extract the contents
#   extract_folder_path = 'D:/OneDrive/Kawa/Nyambut Gawe/BRI/BFLP/DDB/LLM/RM-GPT/chromadb/db'
  # extract_folder_path = '/content/faiss_index'

#   # Open the ZIP file
#   with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
#       # Extract all contents to the specified folder
#       zip_ref.extractall(extract_folder_path)

  # # Supplying a persist_directory will store the embeddings on disk
  # persist_directory = 'D:/OneDrive/Kawa/Nyambut Gawe/BRI/BFLP/DDB/LLM/RM-GPT/chromadb/db'
  # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

  # # Now we can load the persisted database from disk, and use it as normal.
  # docsearch = Chroma(persist_directory=persist_directory,
  #                   embedding_function=embeddings)

  """DIBAWAH INI JIKA MENGGUNAKAN FAISS INDEX"""
  pickle_path = r"D:\OneDrive\Kawa\Nyambut Gawe\BRI\BFLP\DDB\LLM\RM-GPT\faiss_index\faiss_index_reChunk"
  embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
  local_vector_database = FAISS.load_local(pickle_path, embeddings)
  retriever = local_vector_database.as_retriever(search_kwargs={"k": 10})
  
  # Test the knowledge retriever
  llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
  knowledge_base = RetrievalQA.from_chain_type(
      llm=llm, chain_type="stuff", retriever=retriever # docsearch.as_retriever()  (if chromadb)
  )

  return knowledge_base

def get_tools(knowledge_base):
    tools = [
        Tool(
            name="Credit Card BRI Knowledge",
            func=knowledge_base.run,
            description="useful for when you need to answer all questions about credit card product and credit card product recommendation",
        )
    ]

    return tools