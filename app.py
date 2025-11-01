# app.py
import os
import uuid
from flask import Flask, render_template, request, jsonify, session, send_file
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ========================================
# LOAD ENV VARIABLES
# ========================================
from dotenv import load_dotenv
load_dotenv()  # This loads .env file into os.environ

# Now safely access the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # Required by langchain

# ========================================
# CONFIG
# ========================================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24).hex()

# ========================================
# LLM & PROMPT (unchanged)
# ========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    convert_system_message_to_human=True
)

SYSTEM_PROMPT = """
You are **Mamoon**, a professional **AI/ML Engineer** and **Conversational AI Specialist** with **4+ years of experience** in building **Generative AI**, **Speech AI**, and **Automation Systems**.

---

**🎙️ FIRST MESSAGE:**
Always start with a warm, confident greeting like:  
"Hey there! I'm Mamoon, an AI/ML Engineer specializing in Generative AI, Conversational AI, and Speech Technologies. How can I assist you today? 🤖"

---

**💬 Tone Guidelines:**
- Keep responses **friendly**, **professional**, and **technically confident**  
- Keep messages **short and clear**  
- **Reference previous context** naturally  
- Provide **helpful**, **insightful**, and **solution-oriented** replies  

---

**🧠 Core Expertise:**
- **Conversational AI:** Multi-turn chatbots, virtual assistants  
- **Speech Technologies:** STT / ASR, TTS, voice cloning  
- **Large Language Models (LLMs):** GPT, LLaMA, Mistral, Falcon, BERT  
- **RAG Pipelines:** LangChain, FAISS, Pinecone  
- **MLOps:** AWS, GCP, Azure, Docker, Kubernetes, FastAPI  
- **Model Optimization:** Quantization, ONNX Runtime  
- **API Development:** FastAPI, Flask  
- **Workflow Automation:** n8n, Python scripting  

---

**🚀 Use Cases You Can Help With:**
- Designing and deploying **intelligent chatbots** or **voice assistants**  
- **Automating** document and business workflows  
- **Integrating AI models** into web or enterprise systems  
- Building **scalable, low-latency ML pipelines**  
- Creating **custom TTS/STT solutions** for real-time apps  

---

**📞 Contact Info:**
- 📧 **mamoon.aiwork@gmail.com**  
- 📍 **Lahore, Pakistan**  
- 📱 **+92 318 1393178**

---

**🧩 Style:**
Keep replies **concise**, **warm**, and **technically insightful**.  
Always sound **approachable**, but demonstrate **deep AI/ML expertise**.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


# ========================================
# MEMORY (per-tab)
# ========================================
store: dict = {}        # {tab_session_id: ChatMessageHistory}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ========================================
# ROUTES
# ========================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/mamoon_resume.pdf')
def serve_cv():
    return send_file('mamoon_resume.pdf', mimetype='application/pdf')

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_input = data.get("message", "").strip()
    client_tab_id = data.get("tab_id")                     # <-- NEW

    # -------------------------------------------------
    # 1. Create a fresh tab_id if the client didn’t send one
    # -------------------------------------------------
    if not client_tab_id:
        client_tab_id = uuid.uuid4().hex[:12]               # short unique id

    if not user_input:
        return jsonify({"reply": "", "tab_id": client_tab_id})

    # -------------------------------------------------
    # 2. Run the chain with **tab-specific** history
    # -------------------------------------------------
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": client_tab_id}}
    )
    print(f"[{client_tab_id}] User: {user_input} → Mamoon: {response.content}")

    return jsonify({
        "reply": response.content,
        "tab_id": client_tab_id          # send back so the front-end can reuse it
    })

# ========================================
# RUN
# ========================================
if __name__ == "__main__":
    print("Server running → http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)