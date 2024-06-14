import torch
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceInferenceAPI  
from llama_index.core import Settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt,completion_to_prompt
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import PromptTemplate
import warnings
warnings.filterwarnings("ignore")

os.environ["HUGGINGFACEHUB_API_TOKEN"]  = "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"

# print("Please provide pdf file path:")
dir = "summarizer/stanford text/"
documents = SimpleDirectoryReader(input_dir=dir).load_data()

query_str = str(input("\n\n\nQuestion:\n\n"))

query_wrapper_prompt = PromptTemplate(
    "As a Q/A bot, your job is to answer questions and the citations used based on the content of the document provided to you."
    "Here is a question for you:\n\n"
    "### Question:\n{query_str}\n\n### Answer:"
)

llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-alpha")


embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
)

Settings.llm = llm
Settings.node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=5,
    window_metadata_key="window",
    original_text_metadata_key="original_text").get_nodes_from_documents(documents)
Settings.text_splitter = SentenceSplitter(chunk_size=128,chunk_overlap=20)
Settings.embed_model = embed_model

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=20,
    verbose=True,
    response_mode="tree_summarize",
    node_postprocessor=[MetadataReplacementPostProcessor("window")])
response = query_engine.query(query_wrapper_prompt.format(query_str=query_str))
print(response)
print("\n")

