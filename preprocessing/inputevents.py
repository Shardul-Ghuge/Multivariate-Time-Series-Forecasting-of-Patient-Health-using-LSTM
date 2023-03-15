import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

HOSP_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/mimic-iv-2.1/hosp/"
ICU_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/mimic-iv-2.1/icu/"
generate_images = False

#For the inputevents dataset, We select only the patients from admissions_processed dataframe.

admissions = pd.read_csv("processed_data/admissions_processed.csv")
inputs = pd.read_csv(ICU_DIR + "inputevents.csv")

#Restrict the dataset to the previously selected admission ids only.
adm_ids = list(admissions["hadm_id"])
inputs = inputs.loc[inputs["hadm_id"].isin(adm_ids)]
# only keep the columns of interest
inputs = inputs[["subject_id","hadm_id","starttime","endtime","itemid","amount","amountuom","rate","rateuom","patientweight","ordercategorydescription"]]

print("Number of patients remaining in the database: ", inputs["subject_id"].nunique())

#We load the d_items dataframe which contains the name of the itemid. And we merge both tables together.
item_id = pd.read_csv(ICU_DIR + "d_items.csv")
item_id = item_id[["itemid","label"]]

#We merge the name of the item administrated.
inputs = pd.merge(inputs,item_id,on="itemid")

print("Number of patients remaining in the database after itemid merge: ")
print(inputs["subject_id"].nunique())

#For each item, evaluate the number of patients who have been given this item.
pat_for_item= inputs.groupby("label")["subject_id"].nunique()
#Order by occurence and take the 33 best (the ones with the most patients)
frequent_labels = pat_for_item.sort_values(ascending=False)[:50]
print("frequent labels size: ", len(frequent_labels))

#Only select specific labels for the inputs.
#list of retained inputs :
retained_list=["Albumin 5%","Dextrose 5%","Lorazepam (Ativan)","Calcium Gluconate","Midazolam (Versed)","Phenylephrine",
"Furosemide (Lasix)","Hydralazine","Norepinephrine","Magnesium Sulfate","Nitroglycerin","Insulin - Glargine","Insulin - Humalog",
"Insulin - Regular","Heparin Sodium","Morphine Sulfate","Potassium Chloride","Packed Red Blood Cells","Gastric Meds","D5 1/2NS","LR",
"K Phos","Solution","Sterile Water","Metoprolol","Piggyback","OR Crystalloid Intake","OR Cell Saver Intake","PO Intake","GT Flush",
"KCL (Bolus)","Magnesium Sulfate (Bolus)"]
inputs = inputs.loc[inputs["label"].isin(retained_list)].copy()

#----------------------------------CLEANING THE INPUT DATA------------------------
#1) Clean the amount column: Verification that all input labels have the same amounts units.
inputs.groupby("label")["amountuom"].value_counts()

##### Cleaning the Cefazolin (remove the ones that are not in dose unit)
inputs=inputs.drop(inputs.loc[(inputs["itemid"]==225850) & (inputs["amountuom"]!="dose")].index).copy()
#Cleaning the Cefepime (remove the non dose)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Cefepime") & (inputs["amountuom"]!="dose")].index).copy()
#Cleaning the Ceftriaxone (remove the non dose)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Ceftriaxone") & (inputs["amountuom"]!="dose")].index).copy()
#Cleaning the Ciprofloxacin (remove the non dose)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Ciprofloxacin") & (inputs["amountuom"]!="dose")].index).copy()
#Cleaning the Famotidine (Pepcid) (remove the non dose)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Famotidine (Pepcid)") & (inputs["amountuom"]!="dose")].index).copy()
#Cleaning the Fentanyl (Concentrate) (remove the non mg)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Fentanyl (Concentrate)") & (inputs["amountuom"]!="mg")].index).copy()
inputs.loc[(inputs["label"]=="Fentanyl (Concentrate)") & (inputs["amountuom"]=="mg"),"amount"]*=1000
inputs.loc[(inputs["label"]=="Fentanyl (Concentrate)") & (inputs["amountuom"]=="mg"),"amountuom"]="mcg"
#Cleaning the Heparin Sodium (Prophylaxis) (remove the non dose)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Heparin Sodium (Prophylaxis)") & (inputs["amountuom"]!="dose")].index).copy()
#Cleaning the Hydromorphone (Dilaudid) (remove the non mg)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Hydromorphone (Dilaudid)") & (inputs["amountuom"]!="mg")].index).copy()
#Cleaning the Magnesium Sulfate (remove the non grams)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Magnesium Sulfate") & (inputs["amountuom"]!="grams")].index).copy()
#Cleaning the Propofol (remove the non mg)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Propofol") & (inputs["amountuom"]!="mg")].index).copy()
#Cleaning the Metoprolol (remove the non mg)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Metoprolol") & (inputs["amountuom"]!="mg")].index).copy()
#Cleaning the Piperacillin/Tazobactam (Zosyn) (remove the non dose)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Piperacillin/Tazobactam (Zosyn)") & (inputs["amountuom"]!="dose")].index).copy()
#Cleaning the Metronidazole (remove the non dose)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Metronidazole") & (inputs["amountuom"]!="dose")].index).copy()
#Cleaning the Ranitidine (Prophylaxis)(remove the non dose)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Ranitidine (Prophylaxis)") & (inputs["amountuom"]!="dose")].index).copy()
#Cleaning the Vancomycin (remove the non dose)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Vancomycin") & (inputs["amountuom"]!="dose")].index).copy()
#Cleaning the Fentanyl. Put the mg to mcg 
inputs.loc[(inputs["itemid"]==221744) & (inputs["amountuom"]=="mg"),"amount"]*=1000
inputs.loc[(inputs["itemid"]==221744) & (inputs["amountuom"]=="mg"),"amountuom"]="mcg"
#Cleaning of the Pantoprazole (Protonix)
    #divide in two (drug shot or continuous treatment and create a new item id for the continuous version)
inputs.loc[(inputs["itemid"]==225910) & (inputs["ordercategorydescription"]=="Continuous Med"),"label"]="Pantoprazole (Protonix) Continuous"
inputs.loc[(inputs["itemid"]==225910) & (inputs["ordercategorydescription"]=="Continuous Med"),"itemid"]=2217441
#remove the non dose from the drug shot version
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Pantoprazole (Protonix)") & (inputs["amountuom"]!="dose")].index).copy()

#Verification that all input labels have the same units.
inputs.groupby("label")["amountuom"].value_counts()

#2) Cleaning Rates
#Cleaning of Dextrose 5%  (remove the non mL/hour)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Dextrose 5%") & (inputs["rateuom"]!="mL/hour")].index).copy()
#Cleaning of Magnesium Sulfate (Bolus)  (remove the non mL/hour)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Magnesium Sulfate (Bolus)") & (inputs["rateuom"]!="mL/hour")].index).copy()
#Cleaning of NaCl 0.9% (remove the non mL/hour)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="NaCl 0.9%") & (inputs["rateuom"]!="mL/hour")].index).copy()
#Cleaning of Piggyback (remove the non mL/hour)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Piggyback") & (inputs["rateuom"]!="mL/hour")].index).copy()
#Cleaning of Packed Red Bllod Cells (remove the non mL/hour)
inputs=inputs.drop(inputs.loc[(inputs["label"]=="Packed Red Blood Cells") & (inputs["rateuom"]!="mL/hour")].index).copy()


#Check if a single unit per drug
inputs.groupby("label")["rateuom"].value_counts()

#split entries that are spread in time
inputs['starttime'] = pd.to_datetime(inputs["starttime"], format='%Y-%m-%d %H:%M:%S')
inputs['endtime'] = pd.to_datetime(inputs["endtime"], format='%Y-%m-%d %H:%M:%S')
inputs["duration"] = inputs['endtime'] - inputs['starttime']


duration_split_hours = 2
to_sec_fact = 3600*duration_split_hours

#split data set in four.

#The first dataframe contains the entries with no rate but with extended duration inputs (over 0.5 hour)
df_temp1=inputs.loc[(inputs["duration"]>timedelta(hours=duration_split_hours)) & (inputs["rate"].isnull())].copy().reset_index(drop=True)
#The second dataframe contains the entries with no rate and low duration entries (<0.5hour)
df_temp2=inputs.loc[(inputs["duration"]<=timedelta(hours=duration_split_hours)) & (inputs["rate"].isnull())].copy().reset_index(drop=True)
#The third dataframe contains the entries with a rate and extended duration inputs (over 0.5 hour)
df_temp3=inputs.loc[(inputs["duration"]>timedelta(hours=duration_split_hours)) & (inputs["rate"].notnull())].copy().reset_index(drop=True)
#The fourth dataframe contains the entries with a rate and low duration entries (< 0.5 hour)
df_temp4=inputs.loc[(inputs["duration"]<=timedelta(hours=duration_split_hours)) & (inputs["rate"].notnull())].copy().reset_index(drop=True)

#Check if split is complete
assert(len(df_temp1.index)+len(df_temp2.index)+len(df_temp3.index)+len(df_temp4.index)==len(inputs.index))


#We then process all of these dfs.
#In the first one, we need to duplicate the entries according to their duration and then divide each entry by the number of duplicates

#We duplicate the rows with the number bins for each injection
df_temp1["repeat"] = np.ceil(df_temp1["duration"].dt.total_seconds()/to_sec_fact).astype(int)
df_new1=df_temp1.reindex(df_temp1.index.repeat(df_temp1["repeat"]))
#We then create the admninistration time as a shifted version of the starttime.
df_new1["charttime"] = df_new1.groupby(level=0)['starttime'].transform(lambda x: pd.date_range(start=x.iat[0],freq=str(60*duration_split_hours)+'min',periods=len(x)))
#We divide each entry by the number of repeats
df_new1["amount"] = df_new1["amount"]/df_new1["repeat"]


# In the third one, we do the same
#We duplicate the rows with the number bins for each injection
df_temp3["repeat"] = np.ceil(df_temp3["duration"].dt.total_seconds()/to_sec_fact).astype(int)
df_new3=df_temp3.reindex(df_temp3.index.repeat(df_temp3["repeat"]))
#We then create the admninistration time as a shifted version of the starttime.
df_new3["charttime"] = df_new3.groupby(level=0)['starttime'].transform(lambda x: pd.date_range(start=x.iat[0],freq=str(60*duration_split_hours)+'min',periods=len(x)))
#We divide each entry by the number of repeats
df_new3["amount"] = df_new3["amount"]/df_new3["repeat"]

df_temp2["charttime"]=df_temp2["starttime"]
df_temp4["charttime"]=df_temp4["starttime"]

#Eventually, we merge all 4 splits into one.
inputs_final = df_new1.append([df_temp2,df_new3,df_temp4])
#The result is a dataset with discrete inputs for each treatment.

inputs_final.to_csv("processed_data/inputevents_processed.csv", index=False)