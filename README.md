The goal is to generate realistic synthetic electronic health records with the same underlying 
latent variable distributions as the real data by training an LSTM on MIMIC-IV Dataset.

The `model` folder includes all the code for the LSTM training, testing and mortality accuracy
prediction. Meanwhile, the `preprocecssing` folder includes all the details needed to convert 
the data from the MIMIC-IV dataset into tensors that can be used to train the LSTM.

Particularly, pay attention to the order in which the files in the `preprocessing` folder are run 
in order to generate similar tensors.

1) First run the `admissions.py`. This will process the Patients database, the admission database and the inputs database. It will ouput `outputevents_processed.csv` and `admissions_processed.csv`.

2) Then run `outputevents.py`. It will process the OUTPUTEVENTS database and produce `outputevents_processed.csv`

3) Run `labevents.py`. It will process the LABEVENTS database and output `labevents_processed.csv`

4) Run `prescriptions.py`. It will process the PRESCRIPTIONS database and output `prescriptions_processed.csv`

5) Those processed tables are merged together in the `complete_tensor_create.py`. This will output: 
`LSTM_tensor_train.csv`, `LSTM_tensor_val.csv`, `LSTM_tensor_test` along with 
`LSTM_death_tags_train.csv`, `LSTM_death_tags_val.csv` and `LSTM_death_tags_test.csv`.

6) Then the `process_tensor_for_LSTM.py` takes in the above csv files and creates tensors of the same name that can
be used to train the LSTM in the `model` folder. This file also does MinMaxScaling to normalize the data.

7) Once the respective tensors are generated, they can be used to train in `mimic_training_testing.py`. 
At the top of the file, the directory where the following tensors is stored can be saved into the path variable `DATA_DIR`.

8) Running the `mimic_training_testing.py` will train and save the best model to the desired directory and also perform
evaluations on the test set. This will also calculate_performance_metrics and output it in the terminal for easy access.
Apart from this, it saves `inputs.npy`, `targets.npy` and `predictions.npy` for the test set which can then be used by 
`visualize_pred.py` to generate plots that compare predictions vs targets

9) The same process is followed by `synthetic_training_testing.py` excpet the fact that it has been tweaked to handle 
synthetic data that is generated via `model/synthetic_data/create_synthetic_data.py`

10) Lastly, both `mimic_training_testing.py` and `synthetic_training_testing.py` make calls to the various LSTM models
(Base_LSTM, Seq2Seq, Seq2Seq_TeacherForcing and Seq2Seq_Attention) defined in the `LSTM.py` file in the `model` folder.

11) If the `mimic_training_testing.py` has been trained on a dataset that has the mortality included in it,
`mortality_accuracy.py` can be called after running the training script to obtain the moratality prediction accuracy as well
