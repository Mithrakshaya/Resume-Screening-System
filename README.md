# 🧠 Resume Screening System (Machine Learning)

## 📌 Description  
This project is a machine learning-based resume screening system that analyzes, filters, and ranks resumes based on a given job description. It helps automate the hiring process by identifying the most relevant candidates efficiently.

---

## 🚀 Features  
- Upload multiple resumes (PDF, DOCX, TXT)  
- Automatic text extraction from resumes  
- Resume classification using machine learning  
- Matching resumes with job description  
- Ranking candidates based on similarity score  
- Selection decision (Selected / Not Selected)  
- Displays model accuracy  

---

## 🛠️ Tech Stack  
- Python  
- Flask  
- Scikit-learn  
- Pandas  
- PyPDF2  
- docx2txt  

---

## ⚙️ How It Works  

1. Load dataset (resume_dataset.csv)  
2. Convert text into numerical format using TF-IDF Vectorizer  
3. Train model using Multinomial Naive Bayes  
4. Upload resumes and enter job description  
5. Extract text from resumes  
6. Predict resume category  
7. Calculate similarity using cosine similarity  
8. Rank resumes and display results  

---

## ▶️ How to Run  

```bash
# Clone the repository
git clone <your-repo-link>

# Navigate to the project folder
cd resume-screening-system

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

---

## 📊 Output  

- Shows:
  - Resume file name  
  - Predicted category  
  - Similarity score  
  - Selection decision  
- Results are sorted based on best match  

---

## 📁 Project Structure  

```
├── app.py
├── resume_dataset.csv
├── uploads/
├── templates/
│   └── matchresume.html
├── README.md
```

---

## 📈 Model Details  

- Vectorization: TF-IDF  
- Algorithm: Multinomial Naive Bayes  
- Similarity Measure: Cosine Similarity  

---

## 🎯 Future Improvements  

- Improve accuracy using advanced ML models  
- Add deep learning/NLP techniques  
- Build better user interface  
- Deploy the project online  

