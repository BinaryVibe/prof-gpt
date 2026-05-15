import streamlit as st
import requests

# 1. Page Config & State Memory (MUST BE AT THE VERY TOP)
st.set_page_config(page_title="Prof GPT", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_docs" not in st.session_state:
    st.session_state.active_docs = []

# 2. Toko Brand Styling (CSS Injection)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; text-transform: uppercase; letter-spacing: -1px; }
    
    [data-testid="stSidebar"] { min-width: 400px !important; max-width: 400px !important; }

    .stButton>button { background-color: #FF5628 !important; color: white !important; border-radius: 0px !important; border: none !important; font-family: 'Syne', sans-serif; font-weight: 800; width: 100%; }
    [data-testid="stFileUploader"] { border: 1px dashed #FF5628; padding: 10px; }

    .stChatMessage { border-radius: 0px !important; padding: 1rem !important; margin-bottom: 10px !important; }
    [data-testid="stChatMessageUser"] { background-color: #FF5628 !important; color: white !important; }
    [data-testid="stChatMessageAssistant"] { background-color: #222222 !important; border: 1px solid #333333; }

    .main-header { font-size: 5rem !important; font-weight: 800; color: #FF5628; line-height: 0.9; margin-bottom: 2rem; }
    .sub-header { color: #F4F4F4; font-family: 'Syne', sans-serif; font-size: 1rem; margin-top: -1.5rem; margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# 3. Header Section
st.markdown('<div class="main-header">PROF<br>GPT.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ACADEMIC NAVIGATOR | Seedhi Baat No Bakwaas</div>', unsafe_allow_html=True)
st.write("---")

# 4. Sidebar - Dashboard & File Upload
with st.sidebar:
    st.markdown('<div style="font-family: \'Syne\', sans-serif; font-size: 2rem; font-weight: 800; color: #FF5628;">DASHBOARD</div>', unsafe_allow_html=True)
    st.write("---")
    
    st.markdown("###  UPLOAD KNOWLEDGE")
    uploaded_files = st.file_uploader("Add Course Materials (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing and chunking documents..."):
                files_to_upload = [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploaded_files]
                try:
                    response = requests.post("http://127.0.0.1:8000/upload", files=files_to_upload)
                    if response.status_code == 200:
                        st.success(f"Successfully processed {len(uploaded_files)} document(s)!")
                        # Add new files to our persistent memory
                        st.session_state.active_docs.extend([f.name for f in uploaded_files])
                    else:
                        st.error(f"Failed to process. Backend returned: {response.status_code}")
                except Exception as e:
                    st.error(f"Could not connect to backend: {e}")
        else:
            st.error("Please select a PDF file first.")
            
    st.write("---")
    
    st.markdown("###  QUICK ACTIONS")
    if st.button("Exam Paper Pattern"):
        st.session_state.messages.append({"role": "user", "content": "What is the general paper pattern for Sessional exams?"})
        st.rerun()

    if st.button("Check Attendance Policy"):
        st.session_state.messages.append({"role": "user", "content": "What is the minimum attendance requirement?"})
        st.rerun()

    if st.button("Fee Challan Guide"):
        st.session_state.messages.append({"role": "user", "content": "How do I generate a fee challan on CU Online?"})
        st.rerun()

    st.write("---")
    
    st.markdown("**Active Knowledge Base:**")
    if st.session_state.active_docs:
        unique_docs = list(set(st.session_state.active_docs))
        for doc in unique_docs:
            st.markdown(f"- `{doc}`")
    else:
        st.markdown("- `Awaiting Uploads...`")

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input Logic
if prompt := st.chat_input("Ask Prof GPT about COMSATS..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# 7. Generate Assistant Response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    current_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("*Thinking...*")
        
        try:
            api_url = "http://127.0.0.1:8000/ask"
            payload = {"query": current_prompt}
            
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer found.")
                intent = data.get("intent", "General")
                sources = data.get("sources", [])
                
                full_response = f"**[{intent}]**\n\n{answer}"
                if sources:
                    source_list = ", ".join(sources)
                    full_response += f"\n\n*Sources: {source_list}*"
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                response_placeholder.error(f"Error: Backend returned {response.status_code}")
                
        except Exception as e:
            response_placeholder.error(f"Could not connect to backend: {str(e)}")