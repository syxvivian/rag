from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader("./faq.pdf")
docs = loader.load()
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
)
documents = text_splitter.split_documents(docs)
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="my_compartment_id",
)
db_dir = 'faq'
db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=db_dir,
)
db.persist()
db = None
