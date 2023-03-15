import pandas as pd
import matplotlib.pyplot as plt

HOSP_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/mimic-iv-2.1/hosp/"
ICU_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/mimic-iv-2.1/icu/"
generate_images = False

#For the inputevents dataset, We select only the patients from admissions_processed dataframe.
admissions = pd.read_csv("processed_data/admissions_processed.csv")
prescriptions = pd.read_csv(HOSP_DIR + "prescriptions.csv")

#Restrict the dataset to the previously selected admission ids only.
adm_ids = list(admissions["hadm_id"])
prescriptions = prescriptions.loc[prescriptions["hadm_id"].isin(adm_ids)]

print("Number of patients remaining in the database: ")
print(prescriptions["subject_id"].nunique())

#This part is for selecting the x most frequent prescriptions. Instead we use the list of prescriptions as in the paper.
n_best = 10
#For each item, evaluate the number of patients who have been given this item.
pat_for_item = prescriptions.groupby("drug")["subject_id"].nunique()
#Order by occurence and take the n_best (the ones with the most patients)
frequent_labels=pat_for_item.sort_values(ascending=False)[:n_best]
print("frequent labels: ", frequent_labels)

#Select only the time series with high occurence.
prescriptions = prescriptions.loc[prescriptions["drug"].isin(list(frequent_labels.index))].copy()

print("Number of patients remaining in the database after n_best outputs: ")
print(prescriptions["subject_id"].nunique())
print("Number of datapoints remaining in the database after n_best outputs: ")
print(len(prescriptions.index))

prescriptions['charttime'] = pd.to_datetime(prescriptions["starttime"], format='%Y-%m-%d %H:%M:%S')
#To avoid confounding labels with labels from other tables, we add "drug" to the name
prescriptions["drug"] = prescriptions["drug"] + " Drug"

prescriptions.to_csv("processed_data/prescriptions_processed.csv", index=False)