import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datetime import timedelta

dir_="~/Desktop/UofT/Thesis/LSTM/mimic-iv-2.1/hosp/"
generate_images = True

#Number of admissions by patient
admissions = pd.read_csv(dir_ +"admissions.csv")
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

#We now add a new column with the duration of each stay.
admissions['admittime']= pd.to_datetime(admissions["admittime"], format='%Y-%m-%d %H:%M:%S')
admissions['dischtime']= pd.to_datetime(admissions["dischtime"], format='%Y-%m-%d %H:%M:%S')
admissions["ELAPSED_TIME"]= admissions["dischtime"] - admissions["admittime"]
admissions["ELAPSED_DAYS"]= admissions["ELAPSED_TIME"].dt.days #Elapsed time in days in ICU
#print(admissions.head(5))

if generate_images:
    plt.figure(2)
    plt.hist(admissions["ELAPSED_DAYS"], bins=np.arange(0, 50)) 
    plt.title('Number of patients with specific duration of admissions in days')
    plt.xlabel('elapsed time in days')
    plt.ylabel('patient count')
    plt.savefig('images/patientCount_vs_elapsedTime_histogram.png')
    #print("Number of patients with specific duration of admissions in days : \n", admissions["ELAPSED_DAYS"].value_counts())




