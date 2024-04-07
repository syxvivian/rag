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

# Example usage
text_from = "Proper waste management is essential to the health of all living systems. Our grantees include Wecyclers, a social enterprise in Lagos, Nigeria, that incentives waste recycling in densely populated urban neighbourhoods, and the WEEE Centre in Nairobi, Kenya, which promotes public awareness of the environmental and health hazards of e-waste and educates the public about reuse, recycling, and safe disposal. Oracle is also a founding sponsor of California Coastal Cleanup Day, which has removed hundreds of thousands of pounds of trash from California waterways each year since 1995"
source_language_code = "en"
target_language_code = "ja"

translated_and_wrapped_text = generateOnly.translate_and_wrap_text(text_from, COMPARTMENT_ID, source_language_code, target_language_code)
print("-----translate to ja-----")
print(translated_and_wrapped_text)

# Example usage
text_from = "適切な廃棄物管理は、すべての生活システムの健全性に不可欠です"
source_language_code = "ja"
target_language_code = "en"

translated_and_wrapped_text = generateOnly.translate_and_wrap_text(text_from, COMPARTMENT_ID, source_language_code, target_language_code)
print("-----translate to en-----")
print(translated_and_wrapped_text)

# Create an instance of GenerativeAiClient
client = generateOnly.GenerativeAiClient(config=config, service_endpoint=GEN_AI_INFERENCE_ENDPOINT)

# Using the second method to generate job description
compartment_id = COMPARTMENT_ID
prompt = """
Generate a job description for a data visualization expert with the following three qualifications only:
1) At least 5 years of data visualization experience
2) A great eye for detail
3) Ability to create original visualizations
"""
model_ocid = GENERATION_MODEL_OCID

# Define parameters to adjust
parameters = {
    'max_tokens': 800
    #,'num_generations': 3
}

# Generate job description using generate_job_description_cohere method and adjust parameters here
generated_text_cohere = client.generate_job_description_cohere(prompt, compartment_id,  model_ocid, parameters)
print("**************************Generate Texts Result whith cohere**************************")
generated_result = generated_text_cohere.inference_response.generated_texts[0].text
# Wrap the translated text by sentence
generated_result_wrapped_text = generateOnly.wrap_text_by_sentence(generated_result,'en')
print(generated_result_wrapped_text)


# Using the second method to generate job description
llam_model_ocid = GENERATION_MODEL_OCID_llam
endpoint = GEN_AI_INFERENCE_ENDPOINT
compartment_id = COMPARTMENT_ID
prompt = """
Generate a job description for a data visualization expert with the following three qualifications only:
1) At least 5 years of data visualization experience
2) A great eye for detail
3) Ability to create original visualizations
"""

# Define parameters to adjust
parameters = {
    'max_tokens': 800,
    'temperature': 1.2,
    'frequency_penalty': 0.1,
    'top_p': 0.8
}

# Generate job description using generate_job_description_llam method and adjust parameters here
generated_text = client.generate_job_description_llam(llam_model_ocid, compartment_id, prompt, parameters)


# Process the generated job description
print("**************************Generate Texts Result whith llam**************************")
generated_result = generated_text.inference_response.choices[0].text
# Wrap the translated text by sentence
generated_result_wrapped_text = generateOnly.wrap_text_by_sentence(generated_result,'en')
print(generated_result_wrapped_text)