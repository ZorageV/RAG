import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings 
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.indices.service_context import ServiceContext
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt,completion_to_prompt
from llama_index.core.node_parser import SentenceSplitter

print("Please Wait till models learns from the data provided")

documents = SimpleDirectoryReader(input_dir="C:/Zorage/ML/Projects/RAG/ragcodes/pdfs").load_data()


llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    #You can also use others LLMs of bigger size by using Quantization through bitsandbytes
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    #model_path="C:\Users\manje\AppData\Local\llama_index\models\mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    #How creative the llm can be while generating responses
    temperature=0.2, 
    max_new_tokens=256,
    context_window=4096,
    generate_kwargs={},
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False,
)

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
)

Settings.llm = llm
Settings.text_splitter = SentenceSplitter(chunk_size=256,chunk_overlap=20)
Settings.embed_model = embed_model
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
while(True):
  print("Any Questions (yes or no only)")
  ans = str(input())
  if(ans=='no'):
    break
  print("Enter your Query:")
  query = str(input())
  print("Generating Response")
  response = query_engine.query(query)
  print(F"Response: \n {response}")


