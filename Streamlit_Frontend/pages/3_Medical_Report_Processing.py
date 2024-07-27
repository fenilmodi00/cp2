import streamlit as st
from transformers import BioGptTokenizer, BioGptForCausalLM, pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import PyPDF2  # Or use pdfminer, depending on your preference
import pytesseract
import torch

st.set_page_config(page_title="Medical Report Analysis", page_icon="🩺",layout="wide")

# Model Initialisation Steps
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
biogpt_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

preprocessing_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Helper Functions
def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file, handling potential errors and OCR for scanned documents.

    Args:
        uploaded_file: The uploaded PDF file object.

    Returns:
        The extracted text as a single string, or an error message if extraction fails.
    """

    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            extracted_text = page.extract_text()

            # Check if the extracted text is empty or very short (likely a scanned page)
            if len(extracted_text.strip()) < 10:
                # Convert page to image (PIL Image object)
                image = page.to_image()

                # Perform OCR using pytesseract
                text += pytesseract.image_to_string(image)
            else:
                text += extracted_text

        return text

    except PyPDF2.utils.PdfReadError:
        return "Error: Invalid PDF file."
    except Exception as e:  # Catch any other unexpected errors
        return f"Error: {e}"
      
def preprocess_text_with_llm(text):
    """Preprocesses medical report text using TinyBERT to remove specific elements."""

    prompt_template = """
    Classify this sentence as:

    *   **Relevant** (POSITIVE) if it contains important medical information about the patient's condition, diagnosis, treatment, etc.
    *   **Irrelevant** (NEGATIVE) if it contains dates, headers, footers, page numbers, or sensitive personal patient information (e.g., names, addresses, phone numbers).

    Sentence: {}
    """

    prompts = [prompt_template.format(sentence) for sentence in text.split(".")]

    classifications = preprocessing_pipeline(prompts)
    preprocessed_text = ".".join([sentence for sentence, label in zip(text.split("."), classifications) if label['label'] == 'POSITIVE']) 

    return preprocessed_text
  
def get_biogpt_response(preprocessed_text, max_length=512, custom_prompt=None):
    """Generates a BioGPT response based on preprocessed medical report text and a custom prompt.

    Args:
        preprocessed_text: The preprocessed medical report text.
        max_length (int): Maximum length of the generated response (in tokens).
        custom_prompt (str, optional): A custom prompt to guide the model's generation.

    Returns:
        The BioGPT generated text response.
    """
    
    # Combine custom prompt (if provided) with the preprocessed text
    if custom_prompt:
        input_text = f"{custom_prompt}\n{preprocessed_text}"
    else:
        input_text = preprocessed_text
    
    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=max_length)

    # Generate output
    output = biogpt_model.generate(inputs, max_length=max_length, num_return_sequences=1)

    # Decode output and return
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# ---- Main Content ----
st.title("Medical Report Analysis")
st.markdown("**Upload your medical report (PDF) and get AI-powered insights and personalized health tips.**")

# File Uploader with styling
uploaded_file = st.file_uploader("", type="pdf", accept_multiple_files=False, key="medical_report")

if uploaded_file is not None:
    with st.spinner("Analyzing your report..."):
            # 1. PDF Parsing: Extract text
        text = extract_text_from_pdf(uploaded_file)

            # 2. Preprocessing
        preprocessed_text = preprocess_text_with_llm(text)
            
            # 3. BioGPT Inference
        custom_prompt = "Analyze the below medical report and extract key insights from it and use it to give personalised dietary and fitness (exercise regime) suggestions based on the report:"  # Example prompt
        biogpt_output = get_biogpt_response(preprocessed_text, custom_prompt=custom_prompt)
        st.success('Medical Report Analysis Successful!', icon="✅")

        # Display Results in an organized manner
    st.subheader("Key Insights and Health Tips!")
    for insight in biogpt_output:
        st.markdown(f"- {insight}")
            
    # Disclaimer with styling
    st.markdown("<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>**Disclaimer:** This is not a medical diagnosis. Always consult a qualified healthcare professional for any health concerns.</div>", unsafe_allow_html=True)
else:
    st.warning("Please upload a PDF file first.")
        

