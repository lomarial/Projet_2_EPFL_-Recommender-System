{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Project 2 Recommender System"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Prerequiesits"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- You have to create a new environment to  be able to run Our model  with the command :  **conda create -n yourenvname python=3.6**\n",
    "- Install Spotlight with the command line : ** conda install -c maciejkula -c pytorch spotlight=0.1.5**\n",
    "- Install pandas with : **pip install pandas**\n",
    "- Install numpy and juyter lab in your new environment with :**pip install numpy , pip install jupyter** \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Run the program "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To run the programm , you just have to enter to the folder when you put the file **run.py** , then you can use a jupyter notebook with run.py or just use your terminal with the command : **Python run.py**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Other files in the project "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the Project you will find other files such:\n",
    "- helpers.py:Provide all functions that help for SGD ,ALS and other algorithms\n",
    "- Project2.ipynb: Contains the implementation and the creation of submission for the SGD ,Baselines and also the analyse of different algorithms with the surprise library.\n",
    "- ModelAlS:the implementation of the ALS algorithm with the creation of the submission\n",
    "- Report:Our project report where we explain all methods used and present the results. \n",
    "-  the data : data_train:the trainning csv file provided by AICROWD , sampleSubmission:The test csv file provided by AICROWd\n",
    "- csv files from our submissions(that give best results)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Final Score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The final score was 1.032,The submission ID:31710 and the team name :We_saw_it_We_recommend_it"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
