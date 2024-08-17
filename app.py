import os
import pickle
from flask import Flask, render_template, request, jsonify
from groq import Groq
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

# Initialize Flask app
app = Flask(__name__)
chat_history = []

# Load the vector store from disk
def load_vector_store(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize the vector store with website content from sitemap
vector_store_path = 'vector_store.pkl'
vector_store = load_vector_store(vector_store_path)
print(f"Loaded vector store from {vector_store_path}")

# Home route to render chat.html
@app.route("/")
def index():
    return render_template('chat.html')

# Route to handle chat interaction
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response,context = get_chat_response(msg, chat_history)
    return   response

# Memory for conversation context
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

# Function to interact with the Groq API
def get_chat_response(prompt, chat_history):
    # Retrieve relevant chunks from the vector store
    retrieved_docs = vector_store.similarity_search(prompt, k=5)

    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Define the prompt with the retrieved context
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f'''You are an assistant specialized in answering questions about INCLUSION ÉCONOMIQUE DES JEUNES (IEJ)and cpa. Respond in the language used by the user—English, French, or Arabic. If you don’t know the answer, simply say "I don't know" and do not fabricate answers. Keep your responses brief, contextual, and accurate. Ensure clarity and conciseness without directly copying from the provided context. Only respond to questions related to IEJ and cpa.

Context: {context}
"'''
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            HumanMessagePromptTemplate.from_template(
                prompt
            ),
        ]
    )

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
    )

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt_template,
        verbose=True,
        memory=memory,
    )

    response = conversation.predict(human_input=prompt)
    message = {'human': prompt, 'AI': response}
    chat_history.append(message)
    for message in chat_history:
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )
    return response,context

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)