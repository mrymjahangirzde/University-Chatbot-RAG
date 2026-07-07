import streamlit as st
import numpy as np
import requests
import chromadb
import uuid
import json
import re
import os

from sentence_transformers import SentenceTransformer

from langchain.schema import Document

from data_loader import (
    create_documents_from_faq
    )


# -----------------------------
# FAQ Dataset Loader
# -----------------------------

def load_faq_dataset():

    with open(
        "data/faq.json",
        "r",
        encoding="utf-8"
    ) as f:

        faq_data = json.load(f)

    return faq_data


# -----------------------------
# Notice Loader
# -----------------------------

def load_notices():

    with open(
        "data/notices.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# -----------------------------
# Return FAQ tp document
# -----------------------------

def create_documents_from_faq():

    faq_data = load_faq_dataset()

    documents = []

    for item in faq_data:

        content = (
            f"سوال: {item['question']}\n"
            f"پاسخ: {item['answer']}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "FAQ Dataset"
                }
            )
        )

    return documents


# -----------------------------
# Normalize Persian
# -----------------------------

def normalize_persian(text):

    if text is None:
        return ""

    text = str(text)

    text = text.strip()

    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")

    text = text.replace("؟", "")
    text = text.replace("،", "")

    text = re.sub(r"\s+", " ", text)

    return text

def extract_keywords(text):

    stopwords = {
        "چیست",
        "چگونه",
        "چطور",
        "است",
        "در",
        "از",
        "برای",
        "چه",
        "کجا",
        "چند",
        "آیا",
        "دانشگاه",
        "دانشکده",
        "ملی",
        "مهارت",
        "شریعتی",
        "تهران",
        "دختران"
    }

    words = normalize_persian(
        text
    ).split()

    keywords = []

    for w in words:

        if len(w) > 2 and w not in stopwords:
            keywords.append(w)

    return keywords


# -----------------------------
# Open Router API
# -----------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "gpt-4o-mini"  
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# -----------------------------
# Embedding Model
# -----------------------------

embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")


# -----------------------------
# ChromaDB
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="university_faqs"
)


# -----------------------------
# Embedding Function
# -----------------------------

def get_embedding(text):

    text = normalize_persian(text)

    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True
    )

    return np.array(embedding)


# -----------------------------
# Open Router Answer Generator
# -----------------------------

def answer_question_with_openrouter(question, context):

    conversation_history = ""

    for message in st.session_state.messages[-6:]:

        conversation_history += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    prompt = f"""
    شما دستیار رسمی دانشکده ملی مهارت دختران دکتر شریعتی تهران هستید.

    وظیفه شما پاسخگویی دقیق و رسمی به سوالات دانشجویان است.

    قوانین:

    قوانین:

    - فقط از اطلاعات موجود در Context استفاده کن.
    - اگر پاسخ دقیق در Context وجود دارد همان متن را بدون خلاصه سازی برگردان.
    - اگر پاسخ در قالب سوال و پاسخ آمده است فقط بخش پاسخ را برگردان.
    - اگر پاسخ دقیق سوال در Context وجود نداشت پاسخ نده.
    - شباهت موضوعی کافی نیست.
    - اگر Context درباره موضوع دیگری بود از آن استفاده نکن.
    - هرگز حدس نزن.
    - هرگز از دانش خودت استفاده نکن.

    در صورت نبود پاسخ دقیق فقط بنویس:

    اطلاعات کافی برای پاسخ به این سوال در پایگاه دانش دانشگاه موجود نیست.
    Context:

    {context}

    Question:

    {question}

    Answer:
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0
    }

    try:

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:

        return f"Error: {e}"


# -----------------------------
# Index Documents
# -----------------------------

def index_documents(chunks):

    for chunk in chunks:

        text = "passage: " + chunk.page_content

        embedding = get_embedding(text)

        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[
                {
                    "source": chunk.metadata.get(
                        "source",
                        "unknown"
                    )
                }
            ]
        )


# -----------------------------
# FAQ Similar Search
# -----------------------------

faq_documents = create_documents_from_faq()

faq_questions = []
faq_answers = []
faq_embeddings = []

for doc in faq_documents:

    parts = doc.page_content.split(
        "پاسخ:",
        maxsplit=1
    )

    question_text = (
        parts[0]
        .replace("سوال:", "")
        .strip()
    )

    answer_text = (
        parts[1]
        .strip()
    )

    faq_questions.append(
        question_text
    )

    faq_answers.append(
        answer_text
    )

    faq_embeddings.append(
        get_embedding(
            "query: " + question_text
        )
    )

faq_embeddings = np.array(
    faq_embeddings
)



def search_faq_semantic(question):

    query_embedding = get_embedding(
        "query: " + question
    )

    similarities = np.dot(
        faq_embeddings,
        query_embedding
    )

    best_idx = np.argmax(
        similarities
    )

    best_score = similarities[
        best_idx
    ]

    query_keywords = set(
        extract_keywords(question)
    )

    matched_keywords = set(
        extract_keywords(
            faq_questions[best_idx]
        )
    )

    keyword_overlap = len(
        query_keywords.intersection(
            matched_keywords
        )
    )

    print(
        f"FAQ Semantic Match -> "
        f"{faq_questions[best_idx]}"
        f" | Score={best_score:.3f}"
    )

    if (
        best_score > 0.90
        and keyword_overlap > 0
    ):

        return {
            "answer": faq_answers[best_idx],
            "matched_question": faq_questions[best_idx],
            "score": float(best_score)
        }

    return None


def extract_answer(doc):

    if "پاسخ:" in doc:

        return doc.split(
            "پاسخ:"
        )[1].strip()

    return doc


def rewrite_query(question):

    prompt = f"""
    سوال دانشجو را به یک سوال رسمی و مناسب برای جستجو در پایگاه دانش دانشگاه تبدیل کن.

    فقط سوال بازنویسی شده را برگردان.

    سوال:

    {question}
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0
    }

    try:

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except:

        return question
    

# -----------------------------
# Retrieval
# -----------------------------

def retrieve_relevant_docs(
    query,
    top_k= 10
):

    query = "query: " + normalize_persian(query)

    query_embedding = get_embedding(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return (
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.markdown("""
<style>

.block-container{
    padding-top:3rem;
}

html, body, [class*="css"]{
    direction:rtl;
    text-align:right;
}

.stChatMessage{
    direction:rtl;
    text-align:right;
}

textarea{
    direction:rtl !important;
    text-align:right !important;
}

input{
    direction:rtl !important;
    text-align:right !important;
}

[data-testid="stSidebar"]{
    right:0;
}

[data-testid="collapsedControl"]{
    right:1rem;
    left:auto;
}

.notice-card{
background:#111827;
border:1px solid #374151;
border-right:5px solid #3b82f6;
border-radius:18px;
padding:18px;
margin-top:12px;
margin-bottom:12px;
}

.notice-title{
font-size:22px;
font-weight:bold;
}

.notice-date{
color:#9ca3af;
margin-top:5px;
}

.notice-summary{
margin-top:15px;
font-size:16px;
line-height:2;
}

</style>
""", unsafe_allow_html=True)


st.image(
    "logo.png",
    width=120
)


st.markdown("""
<div style="
background:#111827;
padding:30px;
border-radius:20px;
border:1px solid #374151;
margin-bottom:20px;
">

<h1 style="
text-align:center;
font-size:28px;
margin-bottom:5px;
">
🎓 دستیار هوشمند دانشکده ملی مهارت
</h1>

<h4 style="
text-align:center;
font-size:18px;
">
دختران دکتر شریعتی تهران
</h4>

<p style="
text-align:center;
">
پاسخگویی هوشمند به سوالات آموزشی،
دانشجویی و اداری
</p>

</div>
""", unsafe_allow_html=True)


if st.button(
    "📢 اطلاعیه‌های جدید",
    use_container_width=True
):
    st.session_state.show_notice = True


# -----------------------------
# Session State Initialization
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "questions_count":0
    }

if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = []

if "prefill" not in st.session_state:
    st.session_state.prefill = None

if "retrieval_logs" not in st.session_state:

    try:

        with open(
            "retrieval_logs.json",
            "r",
            encoding="utf-8"
        ) as f:

            st.session_state.retrieval_logs = json.load(f)

    except:

        st.session_state.retrieval_logs = []

if "memory_entity" not in st.session_state:
    st.session_state.memory_entity = None

if "show_notice" not in st.session_state:
    st.session_state.show_notice = False

# -----------------------------
# Load FAQ Once
# -----------------------------

if collection.count() == 0:

    faq_documents = create_documents_from_faq()

    index_documents(
        faq_documents
    )

    print(
        f"Indexed {len(faq_documents)} FAQ documents."
    )

else:

    print(
        f"Collection already contains {collection.count()} documents."
    )


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("⚙️ پنل مدیریت")

    st.markdown("### 📊 وضعیت سامانه")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "تعداد رکوردهای دانش",
            collection.count()
        )

    conversation_count = len(
        [
            msg
            for msg in st.session_state.messages
            if msg["role"] == "user"
        ]
    )

    with col2:

        st.metric(
            "جلسات پرسش",
            conversation_count
        )

    st.markdown("### 🤖 مدل‌ها")

    st.markdown(
        f"""
        <div style="
            direction:ltr;
            text-align:left;
            background:#1f2937;
            padding:15px;
            border-radius:10px;
            border:1px solid #374151;
        ">
            <b>Embedding Model</b><br>
            multilingual-e5-base
            <br><br>
            <b>LLM</b><br>
            {OPENROUTER_MODEL}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("📈 آمار استفاده")

    st.metric(
        "تعداد سوالات ثبت شده",
        st.session_state.analytics["questions_count"]
    )

    st.markdown("---")

    show_debug = st.toggle(
        "نمایش اطلاعات بازیابی",
        value=False
    )

    if show_debug:

        st.subheader("📚 لاگ بازیابی")

        st.metric(
            "تعداد سوالات ثبت شده",
            st.session_state.analytics["questions_count"]
        )

        st.metric(
            "تعداد لاگ‌های ذخیره شده",
            len(st.session_state.retrieval_logs)
        )

        if len(st.session_state.retrieval_logs) > 0:

            for log in reversed(
                st.session_state.retrieval_logs
            ):

                with st.expander(
                    f"❓ {log['question']}"
                ):

                    for doc, meta, dist in zip(
                        log["contexts"],
                        log["sources"],
                        log["distances"]
                    ):

                        st.markdown(
                            f"""
**منبع:** {meta['source']}

**فاصله معنایی:** {dist:.4f}
"""
                        )

                        st.write(
                            doc[:500]
                        )

                        st.divider()

        else:

            st.info(
                "هنوز لاگی ذخیره نشده است."
            )


# -----------------------------
# Chat Section
# -----------------------------

st.markdown("""
<style>

[data-testid="stChatMessage"]{
    border-radius:15px;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)



# -----------------------------
# Notices
# -----------------------------

if st.session_state.show_notice:

    notices = load_notices()

    for notice in notices:

        with st.container(border=True):

            st.markdown(
                f"### 📢 {notice['title']}"
            )

            st.caption(
                f"📅 {notice['date']}"
            )

            st.warning(
                notice["summary"]
            )

            with st.expander(
                "📄 مشاهده متن کامل اطلاعیه"
            ):

                st.markdown(
                    notice["content"]
                )


st.divider()


# -----------------------------
# FAQ Shortcuts
# -----------------------------

st.info(
    "💡 برای شروع می‌توانید از سوالات پرکاربرد زیر استفاده کنید."
)

st.markdown("""
<h4 style="
margin-top:10px;
color:#9ca3af;
">
🔥 سوالات پرکاربرد
</h4>
""",
unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

question = None

with c1:

    if st.button(
        "💼 مراحل کارآموزی",
        use_container_width=True
    ):

        question = "مراحل اخذ و انجام کارآموزی چیست؟"

with c2:

    if st.button(
        "📑 مراحل پروژه",
        use_container_width=True
    ):

        question = "مراحل اخذ و انجام پروژه چیست؟"

with c3:

    if st.button(
        "🎓 معرفی به استاد",
        use_container_width=True
    ):

        question = "شرایط معرفی به استاد چیست؟"


# -----------------------------
# Chat History
# -----------------------------

for message in st.session_state.messages:

    avatar = "👨‍🎓"

    if message["role"] == "assistant":
        avatar = "🤖"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(
            message["content"]
        )


chat_question = st.chat_input(
    "سوال خود را وارد کنید..."
)

if chat_question:
    question = chat_question


if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.session_state.analytics[
            "questions_count"
        ] += 1

        with st.chat_message(
            "user",
            avatar="👨‍🎓"
        ):
            st.markdown(question)

        # -----------------------------
        # Chroma Retrieval
        # -----------------------------

        faq_result = search_faq_semantic(question)

        if faq_result:

            answer = faq_result["answer"]

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "contexts": [],
                    "metadata": [],
                    "distances": []
                }
            )

            with st.chat_message(
                "assistant",
                avatar="🤖"
            ):
                st.markdown(answer)

        else:

            search_query = rewrite_query(
                question
            )

            print("ORIGINAL:", question)
            print("REWRITTEN:", search_query)

            relevant_contexts, metadata, distances = (
                retrieve_relevant_docs(
                    search_query
                )
            )
            
            filtered_contexts = []
            filtered_metadata = []
            filtered_distances = []

            for doc, meta, dist in zip(
                relevant_contexts,
                metadata,
                distances
            ):

                if dist < 0.70:

                    filtered_contexts.append(doc)
                    filtered_metadata.append(meta)
                    filtered_distances.append(dist)

            relevant_contexts = filtered_contexts
            metadata = filtered_metadata
            distances = filtered_distances

            if len(relevant_contexts) == 0:

                answer = (
                    "اطلاعات کافی برای پاسخ به این سوال در پایگاه دانش دانشگاه موجود نیست."
                )
            
            elif len(relevant_contexts) == 1:

                answer = extract_answer(
                    relevant_contexts[0]
                )

            else: 

                st.session_state.last_retrieval = list(
                    zip(
                        relevant_contexts,
                        metadata,
                        distances
                    )
                )

                st.session_state.retrieval_logs.append(
                    {
                        "question": question,
                        "sources": metadata,
                        "distances": distances,
                        "contexts": relevant_contexts
                    }
                )

                context = ""

                for i, doc in enumerate(
                    relevant_contexts
                ):

                    context += f"""
            سند شماره {i+1}

            {doc}

            --------------------------
            """

                with st.spinner(
                    "در حال جستجو و تولید پاسخ..."
                ):

                    answer = answer_question_with_openrouter(
                        question,
                        context
                    )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "contexts": relevant_contexts,
                    "metadata": metadata,
                    "distances": distances
                }
            )

            with st.chat_message(
                "assistant",
                avatar="🤖"
            ):

                st.markdown(answer)