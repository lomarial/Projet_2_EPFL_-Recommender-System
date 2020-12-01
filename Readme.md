## Project 2 Recommender System

# Prerequiesits

- You have to create a new environment to  be able to run Our model  with the command :  **conda create -n yourenvname python=3.6**
- Install Spotlight with the command line : ** conda install -c maciejkula -c pytorch spotlight=0.1.5**
- Install pandas with : **pip install pandas**
- Install numpy and juyter lab in your new environment with :**pip install numpy , pip install jupyter** 

# Run the program 

To run the programm , you just have to enter to the folder when you put the file **run.py** , then you can use a jupyter notebook with run.py or just use your terminal with the command : **Python run.py**

# Other files in the project 

In the Project you will find other files such:
- helpers.py:Provide all functions that help for SGD ,ALS and other algorithms
- Project2.ipynb: Contains the implementation and the creation of submission for the SGD ,Baselines and also the analyse of different algorithms with the surprise library.
- ModelAlS:the implementation of the ALS algorithm with the creation of the submission
- Report:Our project report where we explain all methods used and present the results. 
-  the data : data_train:the trainning csv file provided by AICROWD , sampleSubmission:The test csv file provided by AICROWd
- csv files from our submissions(that give best results)

# Final Score

The final score was 1.032,The submission ID:31710 and the team name :We_saw_it_We_recommend_it
