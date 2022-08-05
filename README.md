# dp-challenge

## Project structure:

- main.py: Main file that creates document_processing objects, to process the patents and extract the relevant features. After this, loads the pre-trained Random Forest model to classify those patents related to energy, and finally persist the energy patents. 
- document_processing.py: Class that processes the documents. Reads the files from the blob, uncompress the files, and extracts the text content. After this creates a spark Dataframe from the text and extracts the different relevant columns from the patent. Finally, save the processed patents on a temp directory
- train_data.ipynb: Notebook to process the train data and creates the training dataset.
- train_model.py: Python file that reads the training dataset, and trains the Random Forest classifier. 

## Code Deployment:

To perform the code deployment you need to follow these steps:

1. Replace the value of the key 'blob_key' with the connection string value. This variable is in line 13 of the main.py file, inside of the "configs" dict
2. Replace the variable "conn_string" with the value of the connection string in the file train_data.ipynb
3. Upload all the project files to Databricks

## Code execution:

After the code deployment, is necessary to execute the following commands in a different cell in the files:

- document_processing.py:
	- %pip install azure-storage-blob
	- %pip install nltk

- main.py
	- %pip install azure-storage-blob
	- ./document_processing

The commands are commented on in the file.

The model dependencies must be available in Databricks. In case of these dependencies are not, is needed to perform the following steps:

- Run the train_data.ipynb
- Execute the following command in the train_model.py:
	%pip install sklearn
- Run the file train_model.py

After this, the main.py file can be executed.