
import io
import os
import re
import time
from dotenv import load_dotenv
import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import base64
from gtts import gTTS
from pydub import AudioSegment

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

st.set_page_config(
    page_title="Article Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    .title {
        font-size: 3.5em;
        margin-top: -1.5em;
        margin-left: -0.6em;
        text-align: right;
        color: #1E90FF; /* Dodger Blue */
    }
    .version {
        font-size: 0.7em;
        vertical-align: super;
        color: cyan;
    }
    .subtitle {
        font-size: 1.5em;
        font-weight: bold;
        color: cyan;
    }
    .sidebar-section {
        margin-top: 1.5em;
    }
    .footer {
        position: fixed;
        right: 0;
        bottom: 10;
        padding: 10px 25px;
        font-size: 14px;
        color: #696969; /* Dim Gray */
    }
    .large-text-input input {
        font-size: 1.2em !important;
        padding: 0.5em 1em !important;
    }
    .sidebar-video {
        margin-top: -20px;
        margin-bottom: 30px;
        margin-left: -1rem;
        margin-right: -1rem;
        padding: 0;
    }
    .sidebar-video video {
        width: 100%;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

video_path = "Red and Blue Mascot Gaming Studio Free Logo1.mp4"  
if os.path.exists(video_path):
    video_html = f"""
        <div class="sidebar-video">
            <video style="max-width: 100%; height: auto;" autoplay loop muted>
                <source src="data:video/mp4;base64,{get_base64_of_bin_file(video_path)}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
    """
    st.sidebar.markdown(video_html, unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

def get_article_text(urls):
    articles = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = "\n".join([para.get_text() for para in paragraphs])
        articles.append((url, article_text))
    return articles

from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_chunks(articles):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = []
    for url, article in articles:
        article_chunks = text_splitter.split_text(article)
        chunks.extend([{"content": chunk, "source": url} for chunk in article_chunks])
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    texts = [chunk["content"] for chunk in text_chunks]
    metadatas = [{"source": chunk["source"]} for chunk in text_chunks]
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")

template = """
You are a chatbot having a conversation with a human.
Given the following extracted parts of a long document and a question, create a final answer. If you don't know the context, then don't answer the question.

context: \n{context}\n

question: \n{question}\n
Response : """

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)
    chains = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chains

def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.load_local(pickle_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    return faiss_index

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = load_faiss_index("faiss_index")
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})

    source_urls = list(set([doc.metadata.get('source', 'Unknown source') for doc in docs]))
    return response["output_text"], source_urls

def is_valid_url(url):
    pattern = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(pattern, url) is not None

def show_message(placeholder, message, message_type, duration=3):
    if message_type == "warning":
        placeholder.warning(message)
    elif message_type == "error":
        placeholder.error(message)
    elif message_type == "success":
        placeholder.success(message)
    time.sleep(duration)
    placeholder.empty()


    try:
        tts = gTTS(text, lang='en-uk')
        tts.save("answer.mp3")
        st.success("Audio file created successfully.")
        
        # Add a small delay to ensure file is written
        time.sleep(1)
        
        if os.path.exists("answer.mp3"):
            audio = AudioSegment.from_mp3("answer.mp3")
            audio.export("answer.wav", format="wav")
            
            with open("answer.wav", "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
            
            # Clean up files
            os.remove("answer.mp3")
            os.remove("answer.wav")
        else:
            st.error("MP3 file not found after creation.")
    except Exception as e:
        st.error(f"Error in audio generation: {str(e)}")


def generate_audio(response):
    try:
        tts = gTTS(response, lang='en-uk')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Error in audio generation: {str(e)}")
        return None


def main():
    st.markdown('<h1 class="title">Article 📃 Analyzer <span class="version">1.1</span></h1>', unsafe_allow_html=True)
    st.markdown('<marquee scrollamount=16><h3 class="subtitle">Analyze and query multiple articles with ease</h3></marquee>', unsafe_allow_html=True)
    
    main_placeholder = st.empty()
    query_instruction = st.empty()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    user_question = st.text_input("", placeholder="Enter prompt")
    st.markdown("---")

    with st.sidebar:
        st.markdown("---")
        st.sidebar.markdown("<h1 style='font-size: 30px; font-weight: bold; color: cyan; text-align: center;'> Articles' URLs </h1>", unsafe_allow_html=True) 
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        url_list = st.text_area("Enter URLs (one per line)", placeholder="https://example.com/article1\nhttps://example.com/article2") 
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Analyze", key="analyze_button", use_container_width=True):
            urls = url_list.splitlines()
            if not urls:
                show_message(main_placeholder, "Please enter at least one URL.", "warning")
            else:
                invalid_urls = [url for url in urls if not is_valid_url(url)]
                if invalid_urls:
                    show_message(main_placeholder, f"The following URLs are not valid:\n{', '.join(invalid_urls)}\nPlease enter valid URLs.", "error")
                else:
                    with st.spinner("Analyzing Articles..."):
                        articles = get_article_text(urls)
                        text_chunks = get_text_chunks(articles)
                        get_vector_store(text_chunks)
                        show_message(main_placeholder, "Analysis Successful!!", "success")
                        main_placeholder.write("You can now query the chatbot...")

    response = ""
    source_urls = []
    answer_generated = False

    if user_question:
        urls = url_list.splitlines()
        if not urls:
            show_message(main_placeholder, "Please enter at least one URL before querying.", "warning")
        else:
            query_instruction.empty()
            with st.spinner("Fetching Response..."):
                try:
                    response, source_urls = user_input(user_question)
                    st.markdown("### Answer:")
                    st.text_area("", value=response, height=170, disabled=True)
                    
                    # Generate audio for the response
                    audio_bytes = generate_audio(response)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    
                    answer_generated = True
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                answer_generated = True

    
    if answer_generated:
        st.markdown("---")
        if st.button("Regenerate Answer"):
            st.experimental_rerun()

    if source_urls:
        st.markdown("---")
        st.markdown("### Source URLs:")
        for url in source_urls:
            st.markdown(f"- {url}")
            


st.markdown("""
    <style>
        .footer {
            position: fixed;
            right: 10px;
            bottom: 20px;
            padding: 10px 25px;
            font-size:16px;
            color: grey;
        }
    </style>
    <div class="footer">
        &copy; 2024 <a href="https://www.linkedin.com/in/imstorm23203attherategmail/" style="color: cyan;">Arjun Singh Rajput</a> &ensp; | &ensp; Built with 💙 and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main() 
