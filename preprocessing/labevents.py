import pandas as pd
import matplotlib.pyplot as plt

HOSP_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/mimic-iv-2.1/hosp/"
ICU_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/mimic-iv-2.1/icu/"
generate_images = False

#For the inputevents dataset, We select only the patients from admissions_processed dataframe.
admissions = pd.read_csv("processed_data/admissions_processed.csv")
lab = pd.read_csv(HOSP_DIR + "labevents.csv")

#Restrict the dataset to the previously selected admission ids only.
adm_ids = list(admissions["hadm_id"])
lab = lab.loc[lab["hadm_id"].isin(adm_ids)]

print("Number of patients remaining in the database: ")
print(lab["subject_id"].nunique())

#We load the d_labitems dataframe which contains the name of the itemid. And we merge both tables together.
item_id = pd.read_csv(HOSP_DIR + "d_labitems.csv")
item_id = item_id[["itemid","label"]]

lab = pd.merge(lab,item_id,on="itemid")

print("Number of patients remaining in the database after itemid merge: ")
print(lab["subject_id"].nunique())


n_best = 25
#For each item, evaluate the number of patients who have been given this item.
pat_for_item = lab.groupby("label")["subject_id"].nunique()
#Order by occurence and take the n_best best (the ones with the most patients)
frequent_labels = pat_for_item.sort_values(ascending=False)[:n_best]
print("frequent labels: ", frequent_labels)

#Select only the time series with high occurence
lab = lab.loc[lab["label"].isin(list(frequent_labels.index))].copy()

print("Number of patients remaining in the database after n_best outputs: ")
print(lab["subject_id"].nunique())
print("Number of datapoints remaining in the database after n_best outputs: ")
print(len(lab.index))

lab.to_csv("processed_data/labevents_processed.csv", index=False)

