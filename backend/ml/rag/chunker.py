from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_document(text):
    text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
    chunks= text_splitter.split_text(text)
    return chunks 

