# Synthetic Tabular Data Generation

## 1. Introduction
The modern world is running on data and 2.5 quintillion bytes of data is being generated every day. However, *"With great power comes great responsibility"* especially when it comes to protecting people's privacy. This issue can be solved by generating synthetic data which will have a close correlation with the original data. Another issue is that even though more and more data is being generated every day there are certain sectors like medical, health, charity etc. where there is a scarcity of data. So, to help such sectors synthetic data can be generated from the small set of data which is already available. This work is done in collaboration with the subject matter experts at the UK National Innovation Centre of Ageing ([NICA](https://uknica.co.uk/)). This collaboration also allowed interaction with existing innovators in the market such as [AINDO](https://www.aindo.com/).

This project uses The UK Time Diary Study 2014 - 2015 data collected from the UK Data Services which is open source and it includes dairy data of 9388 individuals from randomly selected 4238 households generating a total of 16533 unique records. Data is consist of what each individual in a family does at a certain point in time of a day, they have recorded their activities every 10 minutes throughout the day. It also consists of demographic information about the individuals. All the files are in $spss$ format and all the categorical variables are represented as numeric codes and in the metadata file corresponding labels have been mentioned. 
For generating synthetic tabular data,  Synthetic Data Vault (SDV) is been used. Apart from $spss$ encoding, count encoding has also been used just to see if it makes any difference and to check the reliability of the synthetic data, a correlation-heatmap is generated. Also, a regression model was trained on real and synthetic data to see whether there is any shift in error rate from real data to synthetic data.

## 2. Objective

As the dataset contains sensitive data such as demographic data of people and even though the data has been sanitised (personal details like name, phone number, and email address has been removed) there will always be some kind of privacy issues hence, the main objective of this work is to generate synthetic data from this existing behavioural data using Synthetic Data Vault (SDV) which will protect the privacy of the individuals and at the same time, the data will keep the same statistical consistency as that of real data so that reliable analysis can be carried out or used for training machine learning/deep learning models with reasonable reliability. 

Apart from privacy issues, synthetic data generation solves the problem of having lack of data to train deep learning models as deep neural nets are very data hungry and sectors like health and medical often tend to face the issue of lack of data.

## 3. Steps to execute this Project.
NOTE: Running the whole project might take extended period of time. Training time also depends on your systems computation capabilities.
1. Download/fork/clone the project by clicking [here](https://github.com/roshan-pandey/synthetic-data-gen) and place all **the data files** in ./data/ directory. (Data can be downloaded [here](https://ukdataservice.ac.uk/) if you are a member.)
2. Go to project folder and use requirements.txt file to install the packages required to run this project. Run this command "pip install -r .\requirements.txt"
3. Go to ./src/summary.ipynb and do run all. This will perform all the data manipulation and training of models and save the newly created files in ./data/ directory and models in ./models/ directory.
4. To perform EDA, run all ./src/EDA.ipynb. It will generate all the reports/plots in html format in ./reports/ directory.


## 4. Reports

1. Roshan_Pandey_210113925_CSC8639_Thesis.pdf: This file contains the detailed documentation of the project.
2. Roshan_Pandey_210113925_CSC8639_Poster.pdf: This file contains the overall summary of the project.

## 5. Important Consideration
This project runs perfectly fine on windows machine by following the steps mentioned in section 3. If want to run on different OS, encoding might needs to be changed for text processing.
