import streamlit as st
import json
import numpy as np
import re
import os
from groq import Groq
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from emoji import replace_emoji
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load case base
with open('Data\case_base_with_embeddings.json', 'r') as f:
    case_base = json.load(f)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Technical mappings
TECH_MAPPINGS = {
    "win10": "windows 10", "win 10": "windows 10", "win7": "windows 7", "win 7": "windows 7",
    "win8": "windows 8", "win 8": "windows 8", "win11": "windows 11", "win 11": "windows 11",
    "macos": "mac os", "osx": "mac os", "linux": "linux", "ubuntu": "ubuntu",
    "gpu": "graphics card", "cpu": "processor", "hdd": "hard drive", "ssd": "solid state drive",
    "ram": "memory", "usb": "usb", "hdmi": "hdmi", "vga": "vga", "bios": "bios",
    "mobo": "motherboard", "mb": "motherboard", "psu": "power supply", "nic": "network card",
    "nvidia": "nvidia", "amd": "amd", "intel": "intel",
    "bsod": "blue screen", "crash": "crash", "freeze": "freeze", "hang": "hang",
    "lag": "lag", "stutter": "stutter", "glitch": "glitch", "artifact": "artifact",
    "wifi": "wifi", "wi-fi": "wifi", "ethernet": "ethernet", "lan": "local network",
    "wan": "wide area network", "bluetooth": "bluetooth", "bt": "bluetooth",
    "driver": "driver", "fw": "firmware", "os": "operating system", "app": "application",
    "program": "program", "sw": "software", "hw": "hardware", "gui": "graphical interface",
    "pc": "computer", "laptop": "laptop", "notebook": "laptop", "desktop": "desktop",
    "vram": "video memory", "sata": "sata", "nvme": "nvme", "pcie": "pcie"
}

NOISE_PATTERNS = [
    r"\b(thanks|thank you|thx|please|pls|plz|hello|hi|hey|greetings|dear|regards|cheers|best regards|sincerely)\b",
    r"\b(i think|i believe|in my opinion|imho|maybe|perhaps|probably|possibly)\b",
    r"\b(let me know|get back to me|reply|respond|answer|help me|assist me)\b",
    r"\b(anyone|somebody|someone|everyone|people|folks|guys|experts|gurus)\b",
    r"\b(just|actually|really|basically|literally|simply|honestly|frankly)\b",
]

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""

    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"^Solved[!:\-\s]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text)
    text = re.sub(r"^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*:\s*", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"(see|check|read)\s+(my|this|the)\s+(post|thread|answer)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\n|^)-.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\n|^)\s*thanks?[!.]*\s*$", "", text, flags=re.IGNORECASE|re.MULTILINE)
    text = replace_emoji(text, replace='')
    text = re.sub(r'[:;=8][-^]?[)(DPOp3]', '', text)
    text = re.sub(r'[:;=][-^]?[/\\|]', '', text)
    text = re.sub(r"[:;=][-^]?['\"]", '', text)

    for term, replacement in TECH_MAPPINGS.items():
        text = re.sub(rf"\b{re.escape(term)}\b", replacement, text, flags=re.IGNORECASE)

    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = re.sub(r"\b(0x[0-9a-fA-F]+|Error\s\d+|Err\s?\d+)\b", "[ERROR_CODE]", text)
    text = re.sub(r"\b(v\d+\.\d+|ver\s?\d+\.\d+)\b", "[VERSION]", text)
    text = re.sub(r"\b([A-Za-z]:\\[\\\S|*\S]+\.[A-Za-z]{2,4}|\/\S+\.\w{2,4})\b", "[FILE_PATH]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Embeddings
case_embeddings = np.array([case['embedding'] for case in case_base])

# Retrieval
def retrieve_similar_case(new_description, threshold_high=0.9, threshold_mid=0.6):
    query_clean = clean_text(new_description)
    query_embedding = model.encode([query_clean])
    similarities = cosine_similarity(query_embedding, case_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_case = case_base[best_idx]
    best_score = similarities[best_idx]

    return best_score, best_case

# Revising
def revise_solution_with_groq(user_case, retrieved_case, original_solution):
    prompt = f"""
You are a technical support assistant. A user has reported a new laptop issue.
Here is the new case description:

"{user_case}"

We found a similar case with this description:
"{retrieved_case}"

The existing solution was:
"{original_solution}"

Please revise the solution to fit the new case as accurately as possible. 
- If some steps are irrelevant, remove them.
- If new details should be added, include them.
Respond with ONLY the revised solution, no explanation.
""".strip()

    completion = client.chat.completions.create(
        model="llama3-70b-8192",  # Adjust if model is different
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        stream=True,
    )

    revised_solution = ""
    for chunk in completion:
        revised_solution += chunk.choices[0].delta.content or ""
    
    return revised_solution.strip()


# === Streamlit UI ===
st.title("🔍 Laptop Case-Based Diagnosis")

user_input = st.text_area("Describe your laptop issue:")
if st.button("Find Solution"):
    if not user_input.strip():
        st.warning("Please enter a case description.")
    else:
        score, case = retrieve_similar_case(user_input)

        st.write(f"**Similarity Score:** `{score:.4f}`")
        if score > 0.9:
            st.success("✅ Exact match found.")
            st.write("**Suggested Solution:**")
            st.code(case["solution"])
        elif score > 0.6:
            st.info("📝 Close match. Revising solution with AI assistant...")
            st.write("**Closest Case:**")
            st.markdown(case["question"])
            st.write("**Original Solution:**")
            st.code(case["solution"])
            with st.spinner("Revising with Groq LLaMA-3..."):
                revised = revise_solution_with_groq(user_input, case["question"], case["solution"])
                st.write("**🔧 Revised Solution:**")
                st.code(revised)
        else:
            st.error("❌ No sufficiently similar case found.")
