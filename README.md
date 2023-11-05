
# IT1244 Project README

## How to get started (LSTM)
1) Ensure the [required packages](#Packages) are installed and of the correct version.  

2) Place the *data.parquet* file in the same directory as the main jupyter file.

3) **If using pretrained model(s)**, place the *saved_models/* folder in the same directory as the main jupyter file.

4) Set the parameters to obtain the desired results within the main jupyter file as shown in [Code Parameters](#Code-Parameters).

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

### Packages
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
