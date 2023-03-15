import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

HOSP_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/mimic-iv-2.1/hosp/"
ICU_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/mimic-iv-2.1/icu/"
generate_images = True

#Number of admissions by patient
admissions = pd.read_csv(HOSP_DIR +"admissions.csv")
df = admissions.groupby("subject_id")["hadm_id"].nunique()
if generate_images:
    plt.figure(1)
    plt.hist(df, bins=np.arange(0, 20)) 
    plt.title('Number of patients with specific number of admissions')
    plt.xlabel('admission count')
    plt.ylabel('patient count')
    plt.savefig('images/patientCount_vs_admissionCount_histogram.png')
    #print("Number of patients remaining in the dataframe: ", len(admissions.index))


#No Need to filter out patients with more than 1 admission. We will only consider data for the 1st admission of each patient
"""
#As the majortity of patients only present a single admission, we filter out all the patients with more than 1 admission
subj_ids = list(df[df==1].index) #index of patients with only one visit.
admission_1 = admissions.loc[admissions["subject_id"].isin(subj_ids)] #filter out the patients with more than one visit
print("Number of patients remaining in the dataframe: ", len(admission_1.index))
print(admission_1.head(10))
"""

#We now only keep information about the first admission of each patient
print("Number of patients in the dataframe before sigle admission per patient: ", len(admissions.index)) #431088
admissions = admissions.sort_values(by=['subject_id', 'admittime'], ascending=[True, True]).groupby('subject_id').first().reset_index()
print("Number of patients in the dataframe after sigle admission per patient: ", len(admissions.index)) #180747


#We now add a new column with the duration of each stay.
admissions['admittime'] = pd.to_datetime(admissions["admittime"], format='%Y-%m-%d %H:%M:%S')
admissions['dischtime'] = pd.to_datetime(admissions["dischtime"], format='%Y-%m-%d %H:%M:%S')
admissions["elapsed_time"]= admissions["dischtime"] - admissions["admittime"]
admissions["elapsed_days"]= admissions["elapsed_time"].dt.days #Elapsed time in days in ICU
#print(admissions.head(5))

if generate_images:
    plt.figure(2)
    plt.hist(admissions["elapsed_days"], bins=np.arange(0, 50)) 
    plt.title('Number of patients with specific duration of admissions in days')
    plt.xlabel('Duration in days')
    plt.ylabel('Patient count')
    plt.savefig('images/patientCount_vs_elapsedTime_histogram.png')
    #print("Number of patients with specific duration of admissions in days : \n", admissions["ELAPSED_DAYS"].value_counts())

#Let's now report the death rate as a function of the duration stay in ICU.
deaths_per_duration = admissions.groupby("elapsed_days")["hospital_expire_flag"].sum()
patients_per_duration = admissions.groupby("elapsed_days")["subject_id"].nunique()
death_ratio_per_duration = deaths_per_duration/patients_per_duration
if generate_images:
    plt.figure(3)
    plt.plot(death_ratio_per_duration)
    plt.title("Death Ratio per ICU stay duration")
    plt.xlabel("Duration in days")
    plt.ylabel("Death rate (Number of deaths/Nunber of patients)")
    plt.savefig('images/deathRatio_vs_elapsedTime.png')

#Only preserve patients with stay duration of 48hrs - 8 days after analyzing figure above
admissions = admissions.loc[(admissions["elapsed_days"] >= 2) & (admissions["elapsed_days"] <= 8)]
print("Number of patients in the dataframe after 48hrs - 8 days in ICU: ", len(admissions.index)) #82592

#pick only the selected columns from the admissions dataframe
admissions = admissions[["subject_id", "hadm_id", "admittime", "dischtime", "elapsed_time", "elapsed_days", "race", "hospital_expire_flag"]]

#store the results in a csv file after preprocessing
admissions.to_csv("processed_data/admissions_processed.csv", index=False)
