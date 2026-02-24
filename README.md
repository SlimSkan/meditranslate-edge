# 🏥 MediTranslate Edge

**Translate complex medical text into patient-friendly language**
**🌐 [Try the Live Demo](https://huggingface.co/spaces/SlimSKan/meditranslate-edge)**
**📊 [View Code on Kaggle](https://www.kaggle.com/code/slimkhaled/medgemmainterface)**

Built for the **MedGemma Impact Challenge 2026** - Edge AI Prize

## 🎯 Project Overview

MediTranslate Edge is an offline-capable AI application that translates complex medical terminology into clear, patient-friendly language. Designed specifically for resource-constrained clinical environments without reliable internet access.

## ✨ Features

- 🤖 **Fine-tuned MedGemma AI** - Specialized for medical translation
- 🔍 **Input Validation** - Detects and warns about non-medical text  
- ✅ **Clean Output** - Professional, concise explanations
- 🌐 **Edge Deployment Ready** - Optimized for offline use
- 📱 **Simple Interface** - Easy for healthcare professionals and patients

## 🚀 Try It Live

[Link to deployed app will go here]

## 💻 How It Works

1. User enters medical text (from reports, lab results, etc.)
2. AI validates if text is medical
3. Fine-tuned MedGemma translates to simple language
4. Clean, patient-friendly explanation is displayed

## 🏆 MedGemma Impact Challenge

**Category:** Edge AI Prize

**Focus:** Offline-capable medical AI for resource-limited settings

**Model:** Fine-tuned MedGemma 4B with LoRA adapters

## 📝 Technical Details

- **Base Model:** google/medgemma-4b-it
- **Fine-tuning:** LoRA (Low-Rank Adaptation)
- **Dataset:** 90 medical→simple translation pairs
- **Framework:** Streamlit + PyTorch + Transformers
- **Deployment:** Edge-optimized for offline clinical use

## ⚠️ Disclaimer

This is an educational AI tool. Always consult healthcare professionals for medical advice.

## 👨‍💻 Author

Built by Slim for the MedGemma Impact Challenge 2026

## 📄 License

CC BY 4.0 (as per competition requirements)
