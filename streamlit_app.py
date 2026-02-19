
import streamlit as st
import torch
from transformers import AutoTokenizer,AutoModelForCasualLM
from Peft import PeftModel
from huggingface_hub import login
import re
import os

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title = "MediTranslate Edge",
    page_icon = "🏥",
    layout = "wide"
)

# ----------------------------------
# LOAD MODEL
# ----------------------------------

@st.cache_resource
def load_model():
    """Load fine-tuned MedGemma model"""
    try:
         # Login to HuggingFace (will use secrets in Streamlit Cloud)
         hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))
         if hf_token:
             login(token = hf_token)

        base_model_name = "google/medgemma-4b-it"

        #Load Tokenizer
        with st.spinner("Loading Tokenizer..."):
            tokenizer = Autokenizer.from_pretrained(base_model_name)
            
        # Load base model
        with st.spinner("Loading MedGemma model (this may take 5-10 minutes on first run)..."):
            model = AutoModelForCasualLM.from_pretrained(
                base_model_name,
                device_map = "auto",
                torch_dtype = torch.bfloat16,
                low_cpu_mem_usage = True
            )
    
        # TODO: Load LoRA adapters from HuggingFace Hub
        # For now, we'll use the base model
        # When you upload your adapters to HF, uncomment:
        # adapter_path = "YOUR_USERNAME/medgemma-medical-translator"
        # model = PeftModel.from_pretrained(model, adapter_path)
    
    
    return model, tokenizer

    except Exception as e:
        st.error(f"Error loading model {e}")
        st.stop()

# Load model
with st.spinner("🔄 Loading AI model... Please wait..."):
    model, tokenizer = load_model()
    
# ----------------------------------
# VALIDATION FUNCTION
# ----------------------------------

def clean_response(response):
    """Remove extra text from model output"""
    
    stop_phrases = [
        "Translate the following", "Medical Text:", "Further Explanation",
        "Important Note:", "Great!", "You're welcome", "I am unable",
        "please consult", "educational purposes", "Key improvements",
        "The translation above", "The best answer", "I tried my hardest",
        "perfect", "Have an excellent", "Let us know", "calling back",
        "business hours", "Thank You", "Thank you", "next appointment",
        "We will monitor", "We will start", "Please let us know",
        "Here are some", "We want to ensure", "Provide an example",
        "What is one thing", "How does it relate", "1)", "2)", "3)",
    ]
    
    for phrase in stop_phrases:
        if phrase in response:
            response = response[:response.index(phrase)]
    
    if ".." in response:
        response = response[:response.index("..")] + "."
    
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF\U00002702-\U000027B0]+", flags=re.UNICODE
    )
    response = emoji_pattern.sub('', response)
    
    response = re.sub(r'\*\*.*?\*\*', '', response)
    response = re.sub(r'[\*\-•]\s*', '', response)
    response = ' '.join(response.split())
    response = response.strip().strip('"').strip()
    
    if response and not response.endswith('.'):
        response += '.'
    
    return response

# ----------------------------------
# UI
# ----------------------------------

st.title("🏥 MediTranslate Edge")
st.markdown("""
**Translate complex medical text into patient-friendly language**

This app uses a fine-tuned MedGemma AI model to make medical reports easier to understand.
Built for the MedGemma Impact Challenge - targeting edge deployment for resource-constrained clinical settings.
""")

# Examples in expander
with st.expander("📋 See example medical texts"):
    st.markdown("""
    - Patient presents with acute myocardial infarction, ST-elevation noted on ECG.
    - Lab results indicate elevated TSH levels at 8.5 mIU/L, consistent with primary hypothyroidism.
    - Patient diagnosed with acute deep vein thrombosis in the left lower extremity.
    - Echocardiogram reveals mild mitral valve regurgitation with preserved ejection fraction.
    - Patient presents with bilateral knee pain and stiffness, X-ray confirms degenerative joint disease.
    """)

# Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Medical Text Input")
    
    medical_text = st.text_area(
        "Enter medical text:",
        height=200,
        placeholder="Example: Patient presents with acute myocardial infarction, ST-elevation noted on ECG.",
        help="Enter medical terminology from reports, lab results, or clinical notes"
    )
    
    translate_button = st.button("🔄 Translate to Simple Language", type="primary", use_container_width=True)

with col2:
    st.subheader("✅ Simple Explanation")
    output_placeholder = st.empty()

# Translation
if translate_button:
    if not medical_text.strip():
        st.warning("⚠️ Please enter some medical text first!")
    else:
        # Validate
        is_medical, confidence, matches = is_likely_medical(medical_text)
        
        if not is_medical:
            st.warning(f"⚠️ This doesn't appear to be medical text (found {matches} medical keywords). Results may not be meaningful.")
        
        with st.spinner("🤖 Translating..."):
            prompt = f"""Translate the following medical text into simple, patient-friendly language:

Medical Text: {medical_text}

Simple Explanation:"""
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.3,
                )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            clean = clean_response(response)
            
            with output_placeholder.container():
                st.success("✅ Translation complete!")
                st.write(clean)
                
                # Show confidence
                if is_medical:
                    st.caption(f"✓ Medical text detected (confidence: {confidence:.0%})")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>MediTranslate Edge</strong> - MedGemma Impact Challenge 2026</p>
<p>⚠️ <em>This is an AI-powered educational tool. Always consult healthcare professionals for medical advice.</em></p>
<p>Edge AI deployment optimized for offline clinical environments</p>
</div>
""", unsafe_allow_html=True)
