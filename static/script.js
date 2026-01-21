document.getElementById('predictForm').addEventListener('submit', async function(e) {
  e.preventDefault();

  const payload = {
    gender: document.getElementById('gender').value,
    age: Number(document.getElementById('age').value),
    parent_education: document.getElementById('parent_education').value,
    family_income: Number(document.getElementById('family_income').value),
    attendance_pct: Number(document.getElementById('attendance_pct').value),
    internal_marks: Number(document.getElementById('internal_marks').value),
    assignments_submitted_pct: Number(document.getElementById('assignments_submitted_pct').value),
    previous_grade: document.getElementById('previous_grade').value,
    extracurricular: document.getElementById('extracurricular').value
  };

  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = 'Predicting...';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await response.json();

    if (data.error) {
      resultDiv.innerHTML = `Error: ${data.error}`;
    } else {
      resultDiv.innerHTML = `
        <p class="text-lg font-semibold">Prediction: ${data.prediction ? 'Pass' : 'Fail'}</p>
        <p>Probability: ${(data.probability).toFixed(2)}%</p>
        <p class="text-md text-blue-600 font-medium">Remarks: ${data.remarks}</p>
      `;
    }
  } catch (err) {
    resultDiv.innerHTML = `Error: ${err}`;
  }
});
