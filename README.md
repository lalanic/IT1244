# IT1244 Project README

## Packages
Additional Packages Required:  
pyarrow 14.0.0  
fastparquet 2023.10.1  
pandas 2.1.2  
scikit-learn 1.3.2  
tensorflow 2.14.0  
numpy 1.26.1  
  
You can install the packages using:
`pip install <package_name>`  
Or upgrade using:
`pip install <package_name> --upgrade`

## How to get started (LSTM)
1) Ensure the [required packages](#Packages) are installed and of the correct version.  

2) Place the *data.parquet* file in the same directory as the *main_lstm.ipynp* jupyter file.

3) **If using pretrained model(s)**, place the *saved_models/* folder in the same directory as the *main_lstm.ipynp* jupyter file. Some pretrained models are provided. More information can be found in [Code Parameters](#Code-Parameters).

4) Set the parameters to obtain the desired results within the *main_lstm.ipynp* jupyter file as shown in [Code Parameters](#Code-Parameters).

5) Run the program.

### Code Parameters
#### The following parameters are located at the top of the main jupyter file.

- Setting `is_train_model` to **True** will train a new model. (Overwrites existing model(s))

  Setting `is_train_model` to **False** will use pretrained models in the *saved_models/* folder.
 
- `run_option` has 4 options:
	- `1` will print relative root-mean-square-error for open, high, low, close for a **single company** based on a **company-specific** model. 

	- `2` will print relative root-mean-square-error for open, high, low, close for a **single company** based on a **sector-specific** model.

	- `3` will print an **average** relative root-mean-square error for open, high, low, close for **all companies in a specified sector** based on a **company-specific** model.

	- `4` will print an **average** relative root-mean-square error for open, high, low, close for **all companies in a specified sector** based on a **sector-specific** model.

- `COMPANY_INDEX` selects the company to be used, ranges from 0 to 492.
	- Affects `run_option` `1`.
- `SECTOR_INDEX` selects the sector to be used, ranges from 0 to 10.
	- Affects `run_option` `2`,`3` and `4`.
 	- Pretrained models for `run_option` `3` and `4` for sectors `0`, `1`, `2`, `3`, `4`, `9` and `10` are provided.  

## How to get started for secondary models (Linear Regression, PCR)
1) Ensure the [required packages](#Packages) are installed and of the correct version.  

2) Place the *data.parquet* file in the same directory as the *Other_Models.ipynp* jupyter file.

5) Run the program.

## How to get started for feature visualisation (EDA)
1) Ensure the [required packages](#Packages) are installed and of the correct version.  

2) Place the *data.parquet* file in the same directory as the *EDA.ipynp* jupyter file.

5) Run the program.
