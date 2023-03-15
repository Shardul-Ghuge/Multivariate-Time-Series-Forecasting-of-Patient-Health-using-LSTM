import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv("processed_data/LSTM_tensor_train.csv")
validation = pd.read_csv("processed_data/LSTM_tensor_val.csv")
test = pd.read_csv("processed_data/LSTM_tensor_test.csv")
label_dict = pd.read_csv("processed_data/label_dict.csv")
label_dict = label_dict.drop(label_dict.columns[0], axis=1)
label_dict = label_dict['label'].values.tolist()

def create_tensor(df, tensor_type):
    
    print("creating tensor for " + tensor_type)
    
    # drop the first column
    df = df.drop(df.columns[0], axis=1)

    #create a new dataframe with 82 columns
    new_df = pd.DataFrame(columns=range(82))

    #name the columns from label_dict
    new_df.columns = label_dict

    tsuid = df.groupby(['time_stamp', 'unique_id']).size().reset_index().rename(columns={0:'count'})

    new_df[['time_stamp', 'unique_id']] = tsuid[['time_stamp', 'unique_id']]
    
    #sort the dataframe by increasing unique_id and time_stamp
    new_df = new_df.sort_values(by=['unique_id', 'time_stamp'])

    # replace labels_id with labels
    df['label'] = df['label_code'].apply(lambda x: label_dict[x])

    #fill in the first 82 columns of new datframe based on the label and valuenum columns in the original dataframe 
    def populate_new_df(row):
        unique_id = row['unique_id']
        #get the time_stamp
        time_stamp = row['time_stamp']
        #get the label
        label = row['label']
        #get the value
        value = row['valuenum']
        #add the value to the new dataframe
        new_df.loc[(new_df['unique_id'] == unique_id) & (new_df['time_stamp'] == time_stamp), label] = value
        if row.name % 100000 == 0:
            print(row.name)

    df.apply(populate_new_df, axis=1)
    #print(new_df.head(10))

    #convert the df to a tensor
    num_unique_id = len(new_df.unique_id.unique())
    num_time_stamp = 49 #for range of 0 to 48
    num_features = 82

    #find min unique_id
    min_unique_id = new_df['unique_id'].min()

    #initialize a tensor of shape (num_unique_id, num_time_stamp, num_features) with NaN
    tensor = torch.full((num_unique_id, num_time_stamp, num_features), torch.nan)

    #fill in the tensor with the values from the dataframe with the same unique_id and time_stamp
    def populate_tensor(row):
        unique_id = row['unique_id']
        i = unique_id - min_unique_id
        time_stamp = row['time_stamp']
        tensor[i , time_stamp, :] = torch.tensor(row[:82])
        if row.name % 10000 == 0:
            print(row.name)
    
    new_df.apply(populate_tensor, axis=1)

    #use minmaxscaler to scale the tensor between 0 and 1
    scaler = MinMaxScaler()
    tensor = torch.tensor(scaler.fit_transform(tensor.reshape(-1, num_features)).reshape(num_unique_id, num_time_stamp, num_features))

    #save the tensor
    torch.save(tensor, "processed_tensors/LSTM_tensor_" + tensor_type + ".pt")
    print("done creating tensor for " + tensor_type)


# create tensors in order of increasing size    
create_tensor(test, "test")
create_tensor(validation, "val")
create_tensor(train, "train")

