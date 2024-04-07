import os
from dotenv import load_dotenv
import oci
import textwrap
config = oci.config.from_file()

# Function to wrap text at 80 characters per line while keeping sentences intact
def wrap_text_by_sentence(text, target_language_code, width=80):
    if target_language_code == 'ja':
        splitStr = "。" # Split text into sentences based on period (。)
    else:
        splitStr = ". "
    sentences = text.split(splitStr) 
    wrapped_sentences = []
    for sentence in sentences:
        if sentence:
            # Wrap each sentence with a width of 80 characters
            wrapped_sentence = textwrap.fill(sentence + splitStr, width=width)
            wrapped_sentences.append(wrapped_sentence)
    return "\n".join(wrapped_sentences)


## translate with OCI
# translate methord made with ai_language
def translate_and_wrap_text(text_from, compartment_id, source_language_code, target_language_code):
    # Initialize service client with default config file
    ai_language_client = oci.ai_language.AIServiceLanguageClient(config)
    
    # Send the request to service, some parameters are not required, see API doc for more info
    batch_language_translation_response = ai_language_client.batch_language_translation(
        batch_language_translation_details=oci.ai_language.models.BatchLanguageTranslationDetails(
            documents=[
                oci.ai_language.models.TextDocument(
                    key="1",
                    text=text_from,
                    language_code=source_language_code)],
            compartment_id=compartment_id,
            target_language_code=target_language_code),
        opc_request_id="VUPH6Z9C2LD99QDZBJODaaaaaaaamlgyvhoa4qvxzsoguxpdo62juyzpnaxscwq3n5kaxptrha5ihy4a")
    
    # Get the data from response
    translated_text = batch_language_translation_response.data.documents[0].translated_text
    
    # Wrap the translated text by sentence
    wrapped_text = wrap_text_by_sentence(translated_text,target_language_code)
    
    return wrapped_text


class GenerativeAiClient:
    def __init__(self, config, service_endpoint):
        self.gen_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint=service_endpoint,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240)
        )
        self.generate_text_details = oci.generative_ai_inference.models.GenerateTextDetails()
        self.inference_requests = oci.generative_ai_inference.models.CohereLlmInferenceRequest()
        self.llam_inference_request = oci.generative_ai_inference.models.LlamaLlmInferenceRequest()

    def generate_job_description_cohere(self, prompt, compartment_id, model_id, parameters=None):
      
        # Set parameters values
        self.inference_requests.prompt = prompt
        self.inference_requests.max_tokens = parameters.get('max_tokens', 600)
        self.inference_requests.temperature = parameters.get('temperature', 0.75)
        self.inference_requests.is_stream = parameters.get('is_stream', False)
        self.inference_requests.num_generations = parameters.get('num_generations', 1)

        self.generate_text_details.compartment_id = compartment_id
        self.generate_text_details.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_id)
        self.generate_text_details.inference_request = self.inference_requests

        generate_text_response = self.gen_ai_inference_client.generate_text(self.generate_text_details)

        return generate_text_response.data
    
    def generate_job_description_llam(self, llam_model_ocid, compartment_id, prompt, parameters=None):
        generate_text_detail = oci.generative_ai_inference.models.GenerateTextDetails()
        
        self.llam_inference_request.prompt = prompt

        # Set parameters values
        self.llam_inference_request.max_tokens = parameters.get('max_tokens', 600)
        self.llam_inference_request.temperature = parameters.get('temperature', 1)
        self.llam_inference_request.frequency_penalty = parameters.get('frequency_penalty', 0)
        self.llam_inference_request.top_p = parameters.get('top_p', 0.75)

        generate_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=llam_model_ocid)
        generate_text_detail.inference_request = self.llam_inference_request
        generate_text_detail.compartment_id = compartment_id

        generate_text_response = self.gen_ai_inference_client.generate_text(generate_text_detail)

        return generate_text_response.data


