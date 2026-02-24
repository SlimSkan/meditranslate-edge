import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import re
import os

st.set_page_config(
    page_title="MediTranslate Edge",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        # Try to get token from environment variable FIRST
        hf_token = os.environ.get("HF_TOKEN") or os.getenv("HF_TOKEN")
        
        # Fallback to Streamlit secrets if available
        if not hf_token:
            try:
                hf_token = st.secrets.get("HF_TOKEN")
            except:
                pass
        
        if hf_token:
            login(token=hf_token)
        else:
            st.error("HF_TOKEN not found in environment or secrets!")
            st.stop()
        
        base_model_name = "google/medgemma-4b-it"
        
        with st.spinner("Loading tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        with st.spinner("Loading MedGemma (5-10 min on first run)..."):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
        
        return model, tokenizer
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

with st.spinner("🔄 Loading AI model..."):
    model, tokenizer = load_model()

def is_likely_medical(text):
    """Check if text contains medical keywords"""
    medical_keywords = [
        'patient', 'diagnosis', 'diagnosed', 'condition', 'disease', 'syndrome',
        'infection', 'disorder', 'injury', 'trauma', 'acute', 'chronic',
        'heart', 'cardiac', 'pulmonary', 'respiratory', 'renal', 'hepatic',
        'blood', 'vessel', 'artery', 'vein', 'lung', 'liver', 'kidney',
        'brain', 'neural', 'muscle', 'bone', 'joint', 'tissue',
        'test', 'scan', 'xray', 'x-ray', 'mri', 'ct', 'ultrasound', 'ecg', 'ekg',
        'biopsy', 'lab', 'laboratory', 'result', 'findings', 'examination',
        'pain', 'fever', 'cough', 'nausea', 'vomiting', 'bleeding', 'swelling',
        'shortness of breath', 'fatigue', 'weakness', 'dizziness',
        'medication', 'treatment', 'therapy', 'surgery', 'prescription',
        'dose', 'administered', 'prescribed',
        'elevated', 'decreased', 'normal', 'abnormal', 'positive', 'negative',
        'bilateral', 'unilateral', 'anterior', 'posterior', 'proximal', 'distal',
        'doctor', 'physician', 'nurse', 'surgeon', 'specialist',
        'mmhg', 'mg/dl', 'miu/l', 'bpm', 'units', 'level', 'count',
    ]
    
    text_lower = text.lower()
    matches = sum(1 for keyword in medical_keywords if keyword in text_lower)
    confidence = min(matches / 3, 1.0)
    is_medical = matches >= 2
    
    return is_medical, confidence, matches

def clean_response(response):
    """Clean and format the model's output for better readability"""
    
    # Stop phrases that indicate end of useful content
    stop_phrases = [
        "Translate the following", "Medical Text:", "Thank you",
        "Great!", "Important Note:", "Further Explanation",
        "This is consistent with", "resulting from", "leading to",
        "which can lead", "such as", "etcetera", "along with",
        "including", "like", "regarding", "based on",
    ]
    
    # Cut at first stop phrase
    for phrase in stop_phrases:
        if phrase in response:
            response = response[:response.index(phrase)]
    
    # Limit to first 2-3 sentences for conciseness
    if len(response) > 200:
        sentences = response.split('. ')
        if len(sentences) > 2:
            # Keep first 2 complete sentences
            response = '. '.join(sentences[:2]) + '.'
    
    # Remove trailing dots
    if ".." in response:
        response = response[:response.index("..")] + "."
    
    # Remove emojis
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF\U00002702-\U000027B0]+", flags=re.UNICODE
    )
    response = emoji_pattern.sub('', response)
    
    # Remove markdown formatting
    response = re.sub(r'\*\*.*?\*\*', '', response)
    response = re.sub(r'[\*\-•]\s*', '', response)
    
    # Clean up whitespace
    response = ' '.join(response.split())
    response = response.strip().strip('"').strip()
    
    # Ensure proper ending
    if response and not response.endswith('.'):
        response += '.'
    
    return response

# UI Layout
st.title("🏥 MediTranslate Edge")
st.markdown("**Translate complex medical text into patient-friendly language**")
st.markdown("Built for the MedGemma Impact Challenge 2026 - Edge AI Prize")

with st.expander("📋 See example medical texts"):
    st.markdown("""
    - Patient presents with acute myocardial infarction, ST-elevation noted on ECG
    - Lab results indicate elevated TSH levels at 8.5 mIU/L, consistent with primary hypothyroidism
    - Patient diagnosed with acute deep vein thrombosis in the left lower extremity
    - Echocardiogram reveals mild mitral valve regurgitation with preserved ejection fraction
    - Patient presents with bilateral knee pain and stiffness, X-ray confirms degenerative joint disease
    """)

# Two columns for input and output
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Medical Text Input")
    
    medical_text = st.text_area(
        "Enter medical text:",
        height=200,
        placeholder="Example: Patient presents with acute myocardial infarction, ST-elevation noted on ECG...",
        help="Enter medical terminology from reports, lab results, or clinical notes"
    )
    
    translate_button = st.button("🔄 Translate to Simple Language", type="primary", use_container_width=True)

with col2:
    st.subheader("✅ Simple Explanation")
    output_placeholder = st.empty()

# Translation logic
if translate_button:
    if not medical_text.strip():
        st.warning("⚠️ Please enter some medical text first!")
    else:
        # Validate input
        is_medical, confidence, matches = is_likely_medical(medical_text)
        
        if not is_medical:
            st.warning(f"⚠️ This may not be medical text (found {matches} medical keywords). Translation may not be meaningful.")
        
        # Translate
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
            
            # Display result
            with output_placeholder.container():
                st.success("✅ Translation complete!")
                st.write(clean)
                
                if is_medical:
                    st.caption(f"✓ Medical text detected (confidence: {confidence:.0%})")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>MediTranslate Edge</strong> - MedGemma Impact Challenge 2026</p>
<p>⚠️ <em>This is an AI-powered educational tool. Always consult healthcare professionals for medical advice.</em></p>
<p>Edge deployment optimized for offline clinical environments</p>
</div>
""", unsafe_allow_html=True)

