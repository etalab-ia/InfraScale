import json
from pathlib import Path
import streamlit as st

@st.cache_data
def load_translations():
    translations = {}
    translation_path = Path(__file__).parent / "translations"
    
    for lang_code, lang_file in [("en", "en.json"), ("fr", "fr.json")]:
        try:
            with open(translation_path / lang_file, "r", encoding="utf-8") as f:
                translations[lang_code] = json.load(f)
        except FileNotFoundError:
            pass

    if "en" not in translations:
        translations["en"] = {
            "app_title": "LLM Inference GPU Calculator",
            "app_description": "Estimate the number of GPUs required to serve a Large Language Model (LLM)",
        }
    
    return translations

def get_text(key: str, **kwargs) -> str:
    lang = st.session_state.get("language", "en")
    translations = st.session_state.get("translations", {})
    text = translations.get(lang, {}).get(key, translations.get("en", {}).get(key, key))

    if kwargs:
        try:
            text = text.format(**kwargs)
        except:
            pass
    return text
