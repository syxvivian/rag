from langchain.chains import ConversationChain
from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import oci
import os
from dotenv import load_dotenv

config = oci.config.from_file()
load_dotenv()
SUMMARIZE_MODEL_OCID = os.getenv('SUMMARIZE_MODEL_OCID') 
GEN_AI_INFERENCE_ENDPOINT = os.getenv('GEN_AI_INFERENCE_ENDPOINT') 
COMPARTMENT_ID = os.getenv('COMPARTMENT_ID') 
GENERATION_MODEL_OCID = os.getenv('GENERATION_MODEL_OCID') 
GEN_AI_ENDPOINT = os.getenv('GEN_AI_ENDPOINT') 
GENERATION_MODEL_OCID_llam = os.getenv('GENERATION_MODEL_OCID_llam') 

cid = COMPARTMENT_ID
ep = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
llm = OCIGenAI(
    model_id="cohere.command",
    service_endpoint=ep,
    compartment_id=cid,
    model_kwargs={"temperature": 0.7, "max_tokens": 1000, },
)
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint=ep,
    compartment_id=cid,
)
db_dir = 'faq'
loaded_db = Chroma(persist_directory=db_dir,
                   embedding_function=embeddings
                   )
retriever = loaded_db.as_retriever()
QA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)
print(QA.invoke('How can I contact a customer service representative?'))
