from flask import Flask, render_template, request
import os
import PyPDF2
import docx2txt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


# -------- LOAD DATASET -------- #
df = pd.read_csv("resume_dataset.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Resume'])
y = df['Category']

# -------- TRAIN TEST SPLIT -------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- MODEL -------- #
model = MultinomialNB()
model.fit(X_train, y_train)

# -------- ACCURACY -------- #
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# -------- TEXT EXTRACTION -------- #

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)


def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""


# -------- ROUTES -------- #

@app.route('/')
def home():
    return render_template("matchresume.html", accuracy=round(accuracy, 2))


@app.route('/matcher', methods=['POST'])
def matcher():
    job_description = request.form.get('job_description')
    resume_files = request.files.getlist('resumes')

    results = []

    for file in resume_files:
        if file.filename != "":
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            resume_text = extract_text(path)

            # -------- ML PREDICTION -------- #
            vector = vectorizer.transform([resume_text])
            predicted_category = model.predict(vector)[0]

            # -------- SIMILARITY -------- #
            sim_vectors = vectorizer.transform([job_description, resume_text])
            similarity = cosine_similarity(
                sim_vectors[0:1], sim_vectors[1:2]
            )[0][0]

            # -------- IMPROVED DECISION -------- #
            threshold = 0.5  # adjust (0.4–0.7)

            if similarity >= threshold:
                decision = "Selected"
            elif predicted_category.lower() in job_description.lower():
                decision = "Selected (Category Match)"
            else:
                decision = "Not Selected"

            results.append({
                "filename": file.filename,
                "category": predicted_category,
                "decision": decision,
                "similarity": round(similarity, 2)
            })

    # -------- SORT RESULTS (Best First) -------- #
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    return render_template(
        "matchresume.html",
        message="Results:",
        results=results,
        accuracy=round(accuracy, 2)
    )


# -------- RUN -------- #

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)