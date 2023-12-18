import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message
from dotenv import load_dotenv, find_dotenv
from tempfile import NamedTemporaryFile

# loading environment variables
load_dotenv(find_dotenv(), override=True)

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(uploaded_file):
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Use the appropriate loader based on the file extension
    extension = os.path.splitext(uploaded_file.name)[1]
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(tmp_file_path)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(tmp_file_path)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004
# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if 'uploaded_files_per_tab' not in st.session_state:
    st.session_state['uploaded_files_per_tab'] = {}


def main():
    st.image('logo.png')
    st.subheader('LLM Question-Answering Application 🤖')

    # API Key input
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

    # Tab configuration
    num_tabs = st.sidebar.number_input("Number of Tabs", min_value=1, max_value=10, value=1)

    # Initialize session state for each tab
    for i in range(num_tabs):
        if f'vector_store_{i}' not in st.session_state:
            st.session_state[f'vector_store_{i}'] = None
        if f'chat_history_{i}' not in st.session_state:
            st.session_state[f'chat_history_{i}'] = []

    # Tab interface
    tabs = st.tabs([f"Chat {i+1}" for i in range(num_tabs)])
    for i, tab in enumerate(tabs):
        with tab:
            # Document upload and processing
            uploaded_file = st.file_uploader(f'Upload a file for Chat {i+1}:', type=['pdf', 'docx', 'txt'], key=f'file_uploader_{i}')
            chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, key=f'chunk_size_{i}')
            process_file = st.button('Process File', key=f'process_file_{i}')
            if process_file and uploaded_file:
                data = load_document(uploaded_file)
                chunks = chunk_data(data, chunk_size)
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                vector_store = create_embeddings(chunks)
                st.session_state[f'vector_store_{i}'] = vector_store
                st.success(f'File for Chat {i+1} uploaded, chunked, and embedded successfully.')
                if f'tab_{i}' not in st.session_state['uploaded_files_per_tab']:
                    st.session_state['uploaded_files_per_tab'][f'tab_{i}'] = []
                st.session_state['uploaded_files_per_tab'][f'tab_{i}'].append(uploaded_file.name)

            # Chat interaction
            user_input = st.text_area("Your question:", key=f'input_{i}', height=100)
            submit_button = st.button("Send", key=f'send_{i}')
            st.write(f"Uploaded Files for Chat {i+1}:")
            for file_name in st.session_state['uploaded_files_per_tab'].get(f'tab_{i}', []):
                st.write(file_name)
            if submit_button and st.session_state[f'vector_store_{i}']:
                modified_input = f"{user_input} Answer only based on the text you received as input. Don't search external sources. If you can't answer then return `I DONT KNOW`."
                response = ask_and_get_answer(st.session_state[f'vector_store_{i}'], modified_input, 3)  # 'k' value is set to 3
                st.session_state[f'chat_history_{i}'].append((user_input, response))

                # Display chat history
                chat_history = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state[f'chat_history_{i}']])
                st.text_area("Chat History", value=chat_history, key=f'history_{i}', height=300)


# Run the main function
if __name__ == "__main__":
    main()

# Additional Resources
st.sidebar.markdown("### Additional Resources")
st.sidebar.markdown("[Checkout Avinash's Hugging Face Profile](https://huggingface.co/AvinashPolineni)")
st.sidebar.markdown("[Checkout Avinash's GitHub Profile](https://github.com/polineniavinash)")
st.sidebar.markdown("[Contact Me on LinkedIn](https://linkedin.com/in/avinash-polineni/)")

# Footer
st.markdown("---")
st.caption("© 2023 Avinash Polineni.")