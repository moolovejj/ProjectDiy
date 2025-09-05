import json
from sklearn.feature_extraction.text import CountVectorizer

# โหลดข้อมูล
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

symptoms = [" ".join(item["symptoms"]) for item in data]
diseases = [item["disease"] for item in data]

# แปลงข้อความเป็นเวกเตอร์
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(symptoms)

# บันทึกข้อมูลที่เตรียมไว้
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("prepared_data.pkl", "wb") as f:
    pickle.dump((X, diseases), f)

print("Data prepared and saved.")
