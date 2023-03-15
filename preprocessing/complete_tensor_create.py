import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
import math

HOSP_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/mimic-iv-2.1/hosp/"
ICU_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/mimic-iv-2.1/icu/"
PROCESSED_DATA_DIR = "processed_data/"
generate_images = False

"""
Inputs: labevents_processed, inputevents_processed, outputevents_processed, prescriptions_processed
Outputs: 
1) death_tags.csv: A dataframe with the patient id and the corresponding death label
2) complete_tensor_csv: A dataframe containing all the measurments in tensor version
"""

lab_df=pd.read_csv(PROCESSED_DATA_DIR +"labevents_processed.csv")[["subject_id","hadm_id","charttime","valuenum","label"]]
inputs_df=pd.read_csv(PROCESSED_DATA_DIR +"inputevents_processed.csv")[["subject_id","hadm_id","charttime","amount","label"]]
outputs_df=pd.read_csv(PROCESSED_DATA_DIR +"outputevents_processed.csv")[["subject_id","hadm_id","charttime","value","label"]]
presc_df=pd.read_csv(PROCESSED_DATA_DIR +"prescriptions_processed.csv")[["subject_id","hadm_id","charttime","dose_val_rx","drug"]]

#Process names of columns to have the same everywhere.

#Change the name of amount. valuenum for every table
inputs_df["valuenum"]=inputs_df["amount"]
inputs_df.head()
inputs_df=inputs_df.drop(columns=["amount"]).copy()

#Change the name of amount. valuenum for every table
outputs_df["valuenum"]=outputs_df["value"]
outputs_df=outputs_df.drop(columns=["value"]).copy()

#Change the name of amount. valuenum for every table
presc_df["valuenum"]=presc_df["dose_val_rx"]
presc_df=presc_df.drop(columns=["dose_val_rx"]).copy()
presc_df["label"]=presc_df["drug"]
presc_df=presc_df.drop(columns=["drug"]).copy()

#Tag to distinguish between lab and inputs events
inputs_df["origin"]= "inputs"
lab_df["origin"]= "lab"
outputs_df["origin"]= "outputs"
presc_df["origin"]= "prescriptions"

#merge both dfs.
merged_df1=(inputs_df.append(lab_df)).reset_index()
merged_df2=(merged_df1.append(outputs_df)).reset_index()
merged_df2.drop(columns="level_0",inplace=True)
merged_df=(merged_df2.append(presc_df)).reset_index()

#Check that all labels have different names.
assert(merged_df["label"].nunique()==(inputs_df["label"].nunique()+lab_df["label"].nunique()+outputs_df["label"].nunique()+presc_df["label"].nunique()))
#print(merged_df.head(5))

#Set the reference time as the lowest chart time for each admission.
merged_df['charttime']=pd.to_datetime(merged_df["charttime"], format='%Y-%m-%d %H:%M:%S')
ref_time=merged_df.groupby("hadm_id")["charttime"].min()

merged_df_1=pd.merge(ref_time.to_frame(name="ref_time"),merged_df,left_index=True,right_on="hadm_id")
merged_df_1["time_stamp"]=merged_df_1["charttime"]-merged_df_1["ref_time"]
assert(len(merged_df_1.loc[merged_df_1["time_stamp"]<timedelta(hours=0)].index)==0)

#Create a label code (int) for the labels.
label_dict=dict(zip(list(merged_df_1["label"].unique()),range(len(list(merged_df_1["label"].unique())))))
merged_df_1["label_code"]=merged_df_1["label"].map(label_dict)

merged_df_short=merged_df_1[["hadm_id","valuenum","time_stamp","label_code","origin"]]

#store the label dictionnary in a csv file.
label_dict_df=pd.Series(merged_df_1["label"].unique()).reset_index()
label_dict_df.columns=["index","label"]
label_dict_df["label_code"]=label_dict_df["label"].map(label_dict)
label_dict_df.drop(columns=["index"],inplace=True)
label_dict_df.to_csv(PROCESSED_DATA_DIR + "label_dict.csv")

###-----------------TIME BINNING OF DATA-----------------###
print("Starting time binning of data...")
#First we select the data up to a certain time limit (48 hours)
merged_df_short=merged_df_short.loc[merged_df_short["time_stamp"]<timedelta(hours=48)].copy()
print("Number of patients considered: "+str(merged_df_short["hadm_id"].nunique()))

merged_df_short_binned=merged_df_short.copy()

#Plot the number of "hits" based on the binning. That is, the number of measurements falling into the same bin in function of the number of bins
if generate_images:
    print("Generating image of hits in function of binning...")
    bins_num = range(1,60)
    hits_vec=[]
    for bin_k in bins_num:
        time_stamp_str="time_stamp_bin_"+str(bin_k)
        merged_df_short_binned[time_stamp_str]=round(merged_df_short_binned["time_stamp"].dt.total_seconds()*bin_k/(100*36)).astype(int)
        hits_prop=merged_df_short_binned.duplicated(subset=["hadm_id","label_code",time_stamp_str]).sum()/len(merged_df_short_binned.index)
        hits_vec+=[hits_prop]
    plt.figure(1)   
    plt.plot(bins_num,hits_vec)
    plt.title("Percentage of hits in function of the binning factor")
    plt.xlabel("Number of bins/hour")
    plt.ylabel("% of hits")
    plt.savefig('images/number_of_measurements_falling_into_the_same_bin_as_function_of_number_of_bins.png')
    
#We select the binning factor that gives the best trade-off between the number of bins and the number of hits.
bin_k = 1

print("aggregating time data for lab, input, output and prescriptions")
#We now need to aggregate the data in different ways: lab measurements are averaged, while inputs, outputs, and prescriptions are summed.
merged_df_short["time"] = np.ceil(merged_df_short_binned["time_stamp"].dt.total_seconds()*bin_k/(100*36)).astype(int)

#For lab, we have to average the duplicates.
lab_subset=merged_df_short.loc[merged_df_short["origin"]=="lab",["hadm_id","time","label_code","valuenum"]]
lab_subset["key_id"]=lab_subset["hadm_id"].astype(str)+"/"+lab_subset["time"].astype(str)+"/"+lab_subset["label_code"].astype(str)
#remove all valuenum entries that are null  
lab_subset=lab_subset.loc[~lab_subset["valuenum"].isnull()]
lab_subset["valuenum"]=lab_subset["valuenum"].astype(float)

lab_subset_s=lab_subset.groupby("key_id")["valuenum"].mean().to_frame().reset_index()

lab_subset.rename(inplace=True,columns={"valuenum":"exvaluenum"})
lab_s=pd.merge(lab_subset,lab_subset_s,on="key_id")
print(lab_s.head())
assert(not lab_s.isnull().values.any())

#For inputs, we have to sum the duplicates.
input_subset=merged_df_short.loc[merged_df_short["origin"]=="inputs",["hadm_id","time","label_code","valuenum"]]
input_subset["key_id"]=input_subset["hadm_id"].astype(str)+"/"+input_subset["time"].astype(str)+"/"+input_subset["label_code"].astype(str)
input_subset["valuenum"]=input_subset["valuenum"].astype(float)

input_subset_s=input_subset.groupby("key_id")["valuenum"].sum().to_frame().reset_index()

input_subset.rename(inplace=True,columns={"valuenum":"exvaluenum"})
input_s=pd.merge(input_subset,input_subset_s,on="key_id")
assert(not input_s.isnull().values.any())

#For outpus, we have to sum the duplicates as well.
output_subset=merged_df_short.loc[merged_df_short["origin"]=="outputs",["hadm_id","time","label_code","valuenum"]]
output_subset["key_id"]=output_subset["hadm_id"].astype(str)+"/"+output_subset["time"].astype(str)+"/"+output_subset["label_code"].astype(str)
output_subset["valuenum"]=output_subset["valuenum"].astype(float)

output_subset_s=output_subset.groupby("key_id")["valuenum"].sum().to_frame().reset_index()

output_subset.rename(inplace=True,columns={"valuenum":"exvaluenum"})
output_s=pd.merge(output_subset,output_subset_s,on="key_id")
assert(not output_s.isnull().values.any())

#For prescriptions, we have to sum the duplicates as well.
presc_subset=merged_df_short.loc[merged_df_short["origin"]=="prescriptions",["hadm_id","time","label_code","valuenum"]]
presc_subset["key_id"]=presc_subset["hadm_id"].astype(str)+"/"+presc_subset["time"].astype(str)+"/"+presc_subset["label_code"].astype(str)
#remove valuenum entries that have "-" or "," as range (e.g. 0.5-5)
presc_subset=presc_subset.loc[presc_subset["valuenum"].str.contains(",|-") == False,:]
presc_subset["valuenum"]=presc_subset["valuenum"].astype(float)
presc_subset_s=presc_subset.groupby("key_id")["valuenum"].sum().to_frame().reset_index()

presc_subset.rename(inplace=True,columns={"valuenum":"exvaluenum"})
presc_s=pd.merge(presc_subset,presc_subset_s,on="key_id")
assert(not presc_s.isnull().values.any())

#Now remove the duplicates/
lab_s=(lab_s.drop_duplicates(subset=["hadm_id","label_code","time"]))[["hadm_id","time","label_code","valuenum"]].copy()
input_s=(input_s.drop_duplicates(subset=["hadm_id","label_code","time"]))[["hadm_id","time","label_code","valuenum"]].copy()
output_s=(output_s.drop_duplicates(subset=["hadm_id","label_code","time"]))[["hadm_id","time","label_code","valuenum"]].copy()
presc_s=(presc_s.drop_duplicates(subset=["hadm_id","label_code","time"]))[["hadm_id","time","label_code","valuenum"]].copy()

#We append both subsets together to form the complete dataframe
complete_df1=lab_s.append(input_s)
complete_df2=complete_df1.append(output_s)
complete_df=complete_df2.append(presc_s)

assert(sum(complete_df.duplicated(subset=["hadm_id","label_code","time"])==True)==0) #Check if no duplicates anymore.

# We remove patients with less than 50 observations
id_counts=complete_df.groupby("hadm_id").count()
id_list=list(id_counts.loc[id_counts["time"]<50].index)
complete_df=complete_df.drop(complete_df.loc[complete_df["hadm_id"].isin(id_list)].index).copy()
print("preview of complete_df after time binning")
print(complete_df.head(5))

###------------Dataframe creation for Tensor Decomposition------------------###
print("starting dataframe creation for tensor decomposition")

#1) Creation of a unique index
unique_ids=np.arange(complete_df["hadm_id"].nunique())
np.random.shuffle(unique_ids)
d=dict(zip(complete_df["hadm_id"].unique(),unique_ids))  

Unique_id_dict=pd.Series(complete_df["hadm_id"].unique()).reset_index().copy()
Unique_id_dict.columns=["index","hadm_id"]
Unique_id_dict["unique_id"]=Unique_id_dict["hadm_id"].map(d)
Unique_id_dict.to_csv(PROCESSED_DATA_DIR+"unique_id_dict.csv")
print("done creating unique_id_dict.csv")

unique_id_df = pd.read_csv(PROCESSED_DATA_DIR + "unique_id_dict.csv")
d = dict(zip(unique_id_df["hadm_id"].values,unique_id_df["unique_id"].values))

#2) Death tags data set.
admissions=pd.read_csv(PROCESSED_DATA_DIR + "admissions_processed.csv")
death_tags_s = admissions.groupby("hadm_id")["hospital_expire_flag"].unique().astype(int).to_frame().reset_index()
death_tags_df = death_tags_s.loc[death_tags_s["hadm_id"].isin(complete_df["hadm_id"])].copy()
death_tags_df["unique_id"] = death_tags_df["hadm_id"].map(d)
death_tags_df.sort_values(by="unique_id",inplace=True)
death_tags_df.rename(columns={"hospital_expire_flag":"value"},inplace=True)
death_tags_df.to_csv(PROCESSED_DATA_DIR + "complete_death_tags.csv")
print("done creating complete_death_tags.csv")

#3) Tensor data set
complete_df["unique_id"]=complete_df["hadm_id"].map(d)
complete_tensor_nocov= complete_df[["unique_id","label_code","time"]+["valuenum"]].copy()
complete_tensor_nocov.rename(columns={"time":"time_stamp"},inplace=True)

#4) Normalization of the data (N(0,1))
#Add a column with the mean and std of each different measurement type and then normalize them.
d_mean=dict(complete_tensor_nocov.groupby("label_code")["valuenum"].mean())
complete_tensor_nocov["mean"]=complete_tensor_nocov["label_code"].map(d_mean)
d_std=dict(complete_tensor_nocov.groupby("label_code")["valuenum"].std())
complete_tensor_nocov["std"]=complete_tensor_nocov["label_code"].map(d_std)
complete_tensor_nocov["valuenorm"]=(complete_tensor_nocov["valuenum"]-complete_tensor_nocov["mean"])/complete_tensor_nocov["std"]

#-----------------Creation of the dataset for LSTM operations-----------------#
#We split the data patient-wise and provide imputation methods
print("starting dataframe creation for LSTM")
#Unique_ids of train and test
test_prop=0.1
val_prop=0.2
sorted_unique_ids=np.sort(unique_ids)
train_unique_ids=sorted_unique_ids[:int((1-test_prop)*(1-val_prop)*len(unique_ids))]
val_unique_ids=sorted_unique_ids[int((1-test_prop)*(1-val_prop)*len(unique_ids)):int((1-test_prop)*len(unique_ids))]
test_unique_ids=sorted_unique_ids[int((1-test_prop)*len(unique_ids)):]

#Death tags creation for LSTM
death_tags_train_df=death_tags_df.loc[death_tags_df["unique_id"].isin(list(train_unique_ids))].sort_values(by="unique_id")
death_tags_val_df=death_tags_df.loc[death_tags_df["unique_id"].isin(list(val_unique_ids))].sort_values(by="unique_id")
death_tags_test_df=death_tags_df.loc[death_tags_df["unique_id"].isin(list(test_unique_ids))].sort_values(by="unique_id")

death_tags_train_df.to_csv(PROCESSED_DATA_DIR + "LSTM_death_tags_train.csv")
death_tags_val_df.to_csv(PROCESSED_DATA_DIR + "LSTM_death_tags_val.csv")
death_tags_test_df.to_csv(PROCESSED_DATA_DIR + "LSTM_death_tags_test.csv")
print("done creating LSTM_death_tags")

#Tensor split: Create a segmented tensor (by patients)
complete_tensor_train=complete_tensor_nocov.loc[complete_tensor_nocov["unique_id"].isin(list(train_unique_ids))].sort_values(by="unique_id")
complete_tensor_val=complete_tensor_nocov.loc[complete_tensor_nocov["unique_id"].isin(list(val_unique_ids))].sort_values(by="unique_id")
complete_tensor_test=complete_tensor_nocov.loc[complete_tensor_nocov["unique_id"].isin(list(test_unique_ids))].sort_values(by="unique_id")

complete_tensor_train.to_csv(PROCESSED_DATA_DIR + "LSTM_tensor_train.csv") 
complete_tensor_val.to_csv(PROCESSED_DATA_DIR + "LSTM_tensor_val.csv") 
complete_tensor_test.to_csv(PROCESSED_DATA_DIR + "LSTM_tensor_test.csv") 

#Mean Imputation for traininig set
#Vector containing the mean_values of each dimension.
mean_dims=complete_tensor_train.groupby("label_code")["mean"].mean()
mean_dims.to_csv(PROCESSED_DATA_DIR + "mean_traing_features.csv")
