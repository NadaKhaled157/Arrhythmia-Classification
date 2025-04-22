import pandas as pd
import os

def extract_diagnosis(mat_file, csv_file):
    df_labels = pd.read_csv(csv_file)
    if "Record Name" not in df_labels.columns or "Diagnosis" not in df_labels.columns:
        raise ValueError("CSV file must contain 'Record Name' and 'Diagnosis' columns")
    record_name = os.path.basename(mat_file).replace(".mat", "")
    diagnosis_row = df_labels[df_labels["Record Name"] == record_name]
    if diagnosis_row.empty:
        return f"No diagnosis found for {record_name}"
    return diagnosis_row["Diagnosis"].values[0]

mat_file = r"D:/College/Year Three/Second Term/Medical Equipments 2/ECG Task/Filtered Data/05/054/JS04619.mat"
csv_file = "Decoded_Diagnosis.csv"
diagnosis = extract_diagnosis(mat_file, csv_file)
print(f"Diagnosis: {diagnosis}")
