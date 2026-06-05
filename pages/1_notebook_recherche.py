import streamlit as st
import pathlib
from utils import BASE_DIR

st.set_page_config(page_title="Recherche", page_icon="🔍", layout="wide")
st.title("Notebook de recherche")

# Charger le HTML déjà converti
html_path = BASE_DIR / pathlib.Path("house_hrices_pred.html")

if html_path.exists():
    html_content = html_path.read_text(encoding="utf-8")
    st.components.v1.html(html_content, height=900, scrolling=True)
else:
    st.error(f"Fichier introuvable : {html_path.resolve()}")
    st.info("Vérifiez que `house_hrices_pred.html` est dans le même dossier que `app.py`")