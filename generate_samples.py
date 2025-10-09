# generate_samples.py
from student_performance_predictor import generate_synthetic_data

# Generate datasets
generate_synthetic_data(n=20, filename='sample_students_20.csv')
generate_synthetic_data(n=500, filename='sample_students_500.csv')
generate_synthetic_data(n=1000, filename='sample_students_1000.csv')
