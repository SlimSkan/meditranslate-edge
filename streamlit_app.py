# ==========================================
# FIXED STREAMLIT APP (CORRECTED INDENTATION)
# ==========================================

print("📝 Creating FIXED streamlit_app.py...")

app_code = """import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import re
import os

# Page config
st.set_page_config(
    page_title="MediTranslate Edge",
    page_icon="🏥",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        # Login to HuggingFace
        hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))
        if hf_token:
            login(token=hf_token)
        
        base_model_name = "google/medgemma-4b-it"
        
        # Load tokenizer
        with st.spinner("Loading tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load model
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

# Load model
with st.spinner("🔄 Loading AI model..."):
    model, tokenizer = load_model()

# Validation function
def is_likely_medical(text):
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

# Cleaning function
def clean_response(response):
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
        "[\\U0001F600-\\U0001F64F\\U0001F300-\\U0001F5FF\\U0001F680-\\U0001F6FF"
        "\\U0001F900-\\U0001F9FF\\U00002702-\\U000027B0]+", flags=re.UNICODE
    )
    response = emoji_pattern.sub('', response)
    
    response = re.sub(r'\\*\\*.*?\\*\\*', '', response)
    response = re.sub(r'[\\*\\-•]\\s*', '', response)
    response = ' '.join(response.split())
    response = response.strip().strip('"').strip()
    
    if response and not response.endswith('.'):
        response += '.'
    
    return response

# UI
st.title("🏥 MediTranslate Edge")
st.markdown(\"\"\"
**Translate complex medical text into patient-friendly language**

Built for the MedGemma Impact Challenge 2026 - Edge AI Prize
\"\"\")

# Examples
with st.expander("📋 See example medical texts"):
    st.markdown(\"\"\"
    - Patient presents with acute myocardial infarction, ST-elevation noted on ECG.
    - Lab results indicate elevated TSH levels at 8.5 mIU/L, consistent with primary hypothyroidism.
    - Patient diagnosed with acute deep vein thrombosis in the left lower extremity.
    \"\"\")

# Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Medical Text Input")
    
    medical_text = st.text_area(
        "Enter medical text:",
        height=200,
        placeholder="Example: Patient presents with acute myocardial infarction...",
        help="Enter medical terminology from reports or clinical notes"
    )
    
    translate_button = st.button("🔄 Translate", type="primary", use_container_width=True)

with col2:
    st.subheader("✅ Simple Explanation")
    output_placeholder = st.empty()

# Translation
if translate_button:
    if not medical_text.strip():
        st.warning("⚠️ Please enter some medical text first!")
    else:
        is_medical, confidence, matches = is_likely_medical(medical_text)
        
        if not is_medical:
            st.warning(f"⚠️ This may not be medical text ({matches} medical keywords found)")
        
        with st.spinner("🤖 Translating..."):
            prompt = f\"\"\"Translate the following medical text into simple, patient-friendly language:

Medical Text: {medical_text}

Simple Explanation:\"\"\"
            
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
                
                if is_medical:
                    st.caption(f"✓ Medical text detected ({confidence:.0%} confidence)")

# Footer
st.markdown("---")
st.markdown(\"\"\"
<div style='text-align: center; color: #666;'>
<p><strong>MediTranslate Edge</strong> - MedGemma Impact Challenge 2026</p>
<p>⚠️ <em>Educational tool. Always consult healthcare professionals.</em></p>
</div>
\"\"\", unsafe_allow_html=True)
"""

with open('streamlit_app.py', 'w') as f:
    f.write(app_code)

print("✅ Fixed streamlit_app.py created!")
print(f"📝 File size: {len(app_code):,} characters")
print("\n🔧 Fixed issues:")
print("   ✅ Corrected all indentation")
print("   ✅ Fixed string escaping")
print("   ✅ Cleaned up code structure")

