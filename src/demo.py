import streamlit as st
import requests
from pathlib import Path
import time

st.set_page_config(page_title="Multimodal RAG Engine", layout="wide")

# Check API connection
API_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Display connection status
if not check_api_health():
    st.error("⚠️ API server is not running. Please start it with: `make run-api` or `uvicorn src.api:app --reload`")
    st.stop()

st.title("🔍 Multimodal RAG Engine")
st.markdown("**RAG that understands text, images, tables & charts**")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=['pdf'])
    if st.session_state.get("doc_id"):
        st.caption(f"Active document: `{st.session_state['doc_id']}`")
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files, timeout=300)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Processed {result['elements_processed']} elements")
                    st.session_state['doc_id'] = result.get('doc_id')
                else:
                    st.error(f"Failed to process document: {response.text}")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The document may be too large.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Main chat interface
st.header("💬 Ask Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(f"- {source['type']} from page {source['page']}")

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                payload = {"query": prompt, "top_k": 5}
                if st.session_state.get("doc_id"):
                    payload["doc_id"] = st.session_state["doc_id"]

                response = requests.post(
                    f"{API_URL}/query",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.markdown(result["answer"])
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                    
                    with st.expander("📚 Sources"):
                        for source in result["sources"]:
                            st.write(f"- {source['type']} from page {source['page']}")
                else:
                    error_msg = f"Failed to get response: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            except requests.exceptions.Timeout:
                error_msg = "⏱️ Request timed out. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.markdown("---")
st.markdown("Built on multimodal research from ICISML 2026")
