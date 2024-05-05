# from dotenv import load_dotenv, find_dotenv
import streamlit as st
import time
import requests
import json
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()

# Set Google Search API
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

# Other settings
model_list = ['llama3:8b', 'phi3:3.8b', 'codegemma:7b']
model_info_displayed = ['family', 'parameter_size', 'quantization_level'] # what you want to display on sidebar


# Get model info using API
def fetch_model_info(model_name, keys):
    url = 'http://localhost:11434/api/show'

    # Make the POST request
    response = requests.post(url, json={"name": model_name})

    # Check if the request was successful
    if response.status_code == 200:
        # Process the response data if necessary
        details = response.json()['details']
        model_info = {k: details[k] for k in keys}
    else:
        model_info = {k: '' for k in keys}

    return model_info

def format_keys(keys):
    keys_dict = {}
    for key in keys:
        text_list = key.split('_')
        text_list = [t[0].upper()+t[1:] for t in text_list]

        keys_dict[key] = " ".join(text_list)

    return keys_dict

#########
# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
# Database
vectorstore = Chroma(embedding_function=embedding, persist_directory="./chroma_db")

# Google search API wrapper
search = GoogleSearchAPIWrapper(google_api_key=google_api_key, google_cse_id=google_cse_id)



###### Start Streamlit Code ######

st.title('AI Chat Bot')

# Reset button
reset_button = st.button("Reset Conversation")
if reset_button:
    del st.session_state['messages']
    del st.session_state['context']


with st.sidebar:
    st.subheader('Parameters')
    llm_model = st.selectbox('LLM Model used', model_list)
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=2.0, value=0.3, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    max_new_tokens = st.sidebar.slider('max_new_tokens', min_value=10, max_value=8192, value=300, step=10)

    st.divider()

    st.subheader('Model Metadata')
    model_info = fetch_model_info(llm_model, model_info_displayed)
    formatted_model_info = ""
    for k, f_k in format_keys(model_info_displayed).items():
        formatted_model_info += f"* {f_k}: {model_info[k]}\n"
    
    st.markdown(formatted_model_info)

# Set an instance for LLM model
llm = ChatOllama(
    model=llm_model,
    temperature=temperature,
    top_p=top_p,
    max_new_tokens=max_new_tokens,
)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.context = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


websearch = st.checkbox('Do Web Search')
document = st.checkbox('Use Document as Reference')

# React to user input

# Streamed response emulator
def response_generator(prompt, context):
    global llm, vectorstore, search
    context.append(HumanMessage(content=prompt))

    # TO DO
    # Move websearch checkbox to the bottom of the page
    # Limit the number of output tokens
    # Understand what is the prompt used in background (What time is it now? -> return timedate code)
    # Format text (not be newlined)
    if websearch:
        # Retriever
        web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=vectorstore,
            llm=llm,
            search=search,
        )

        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_research_retriever) 
        result = qa_chain({'question': prompt})
        response = result['answer'] + '\n' + result['sources']
    elif document:
        # TO DO
        # Add RAG process
        pass
    else:
        response = llm.invoke(context).content

    return response

def streamer(response):
    for word in response.split():
        yield word + ' '
        time.sleep(0.05)
    

# Once a text is put into the input widget, the text is assigned as prompt and displayed on the screen
if prompt := st.chat_input('What is up?'):
    # Display user's prompt
    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.session_state.context.append(HumanMessage(content=prompt))

    # Display assistant response
    with st.spinner('Thinking...'):
        response = response_generator(prompt, st.session_state.context)
    
    with st.chat_message('assistant'):
        response = st.write_stream(streamer(response))


    st.session_state.messages.append({'role': 'assistant', 'content': response})
    st.session_state.context.append(AIMessage(content=response))


##########