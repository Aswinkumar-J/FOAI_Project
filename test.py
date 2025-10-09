import requests

url = "http://localhost:5000/predict"
data = {
    "gender": "Male",
    "age": 18,
    "parent_education": "Secondary",
    "family_income": 20000,
    "attendance_pct": 70,
    "internal_marks": 60,
    "assignments_submitted_pct": 65,
    "previous_grade": "C",
    "extracurricular": "Low"
}

r = requests.post(url, json=data)
print(r.json())
