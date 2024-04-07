import ragHandsOn.notebooks.code_rag.bk_genonly.generateOnly as generateOnly
import os
from dotenv import load_dotenv
import oci

config = oci.config.from_file()
load_dotenv()
SUMMARIZE_MODEL_OCID = os.getenv('SUMMARIZE_MODEL_OCID') 
GEN_AI_INFERENCE_ENDPOINT = os.getenv('GEN_AI_INFERENCE_ENDPOINT') 
COMPARTMENT_ID = os.getenv('COMPARTMENT_ID') 
GENERATION_MODEL_OCID = os.getenv('GENERATION_MODEL_OCID') 
GEN_AI_ENDPOINT = os.getenv('GEN_AI_ENDPOINT') 
GENERATION_MODEL_OCID_llam = os.getenv('GENERATION_MODEL_OCID_llam') 

prompt = """
データ可視化の専門家のための仕事の説明を、次の3つの要件のみで生成してください：
1) 少なくとも5年のデータ可視化の経験があること
2) 細部に優れた目を持っていること
3) オリジナルの可視化を作成できる能力があること
"""

text_from = prompt
source_language_code = "ja"
target_language_code = "en"

translated_and_wrapped_text = generateOnly.translate_and_wrap_text(text_from, COMPARTMENT_ID, source_language_code, target_language_code)

# Create an instance of GenerativeAiClient
client = generateOnly.GenerativeAiClient(config=config, service_endpoint=GEN_AI_INFERENCE_ENDPOINT)

# Using the second method to generate job description
compartment_id = COMPARTMENT_ID
model_ocid = GENERATION_MODEL_OCID

# Define parameters to adjust
parameters = {
    'max_tokens': 800
    #,'num_generations': 3
}

# Generate job description using generate_job_description_cohere method and adjust parameters here
generated_text_cohere = client.generate_job_description_cohere(translated_and_wrapped_text, compartment_id,  model_ocid, parameters)
generated_result = generated_text_cohere.inference_response.generated_texts[0].text

text_from = generated_result
source_language_code = "en"
target_language_code = "ja"

translated_text = generateOnly.translate_and_wrap_text(text_from, COMPARTMENT_ID, source_language_code, target_language_code)

# Wrap the translated text by sentence
wrapped_text = generateOnly.wrap_text_by_sentence(translated_text,target_language_code)
    
print(wrapped_text)