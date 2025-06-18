# 💻 Laptop Service Assistant & Diagnosis

This project is a **Case-Based Reasoning (CBR)** system designed to assist users in diagnosing and solving laptop-related issues by referencing previously resolved cases from the [Tom's Guide Laptop Tech Support Forum](https://forums.tomsguide.com/forums/laptop-tech-support.16/).

---

## 🧠 How It Works

1. **Data Collection**: Problem descriptions and their accepted solutions are scraped from solved threads on the forum.
2. **Case Encoding**: These cases are cleaned and encoded using **BERT** to form a searchable case base.
3. **User Input**: The user submits a description of their laptop issue.
4. **Similarity Matching**: The system compares the input to existing cases using **cosine similarity**:
   - **Similarity > 0.90**: Returns the solution from the most similar past case.
   - **0.60 ≤ Similarity ≤ 0.90**: Uses a **LLaMA-based AI model** to adapt a close case's solution to the user's issue.
   - **Similarity < 0.60**: Notifies the user that no sufficiently similar case was found.

---

## ⚙️ Setup Instructions

To run or develop the system locally:

1. Clone the repository  
   ```bash
   git clone https://github.com/Enryu5/CBR-System-Final-Project.git
   cd CBR-System-Final-Project
2. Create and activate a Python virtual environment
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
3. Install dependencies
   ```bash
   pip install -r requirements.txt
4. Configure environment variables\n
   Duplicate .env.example, rename it to .env, and add your Groq API key.
5. Run the application
   ```bash
   streamlit run app.py
