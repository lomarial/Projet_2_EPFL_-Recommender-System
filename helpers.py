# -*- coding: utf-8 -*-
"""some functions for help."""
import torch
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score
torch.set_default_tensor_type('torch.DoubleTensor')
from itertools import groupby

import numpy as np
import scipy.sparse as sp
import csv

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()
def load_csv_data(data_path):
    """Loads data and returns y (ratings), tX (predictions) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=float, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    input_data = x[:, 2:]
    
    with open(data_path, "r") as f:
    	headers = [x.strip() for x in f.readline().split(",")[2:]]


    return y, input_data,headers

def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)
def deal_line(line):
        pos,rating = line.split(',')
        
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)
def seperate_data(path_dataset):
    data = read_txt(path_dataset)[1:]
    # parse each line
    data = [deal_line(line) for line in data]
    return data

def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(",")
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings

def submission(path,pred,name):
    
        
    with open(path, "r") as fin:
        with open(name, "w") as fout:
            fields = ['Id', 'Prediction']
            reader = csv.DictReader(fin, fieldnames=fields)
            writer = csv.DictWriter(fout,delimiter=",", fieldnames=fields)
            writer.writeheader()
            next(reader)   # Skip the column header
            for ind,record in enumerate(reader):
                record['Prediction'] = int(pred[ind])
                writer.writerow(record)
    
def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)
def split_(dataf, column='Id'):
    """

    -param 
    *dataf: The dataframe that willbe splitted using pandas 
    *column: the column that contans the data to split
    return 
    *dataframe containing user_ids, movie_ids and ratings
    """
    datafout = dataf[column].str.split('(\d+)([A-z]+)(\d+)', expand=True)
    datafout = datafout.loc[:, [1, 3]]   #take only the movie ID and the user ID(without r and c)
    datafout.rename(columns={1: 'userid', 2: 'y', 3: 'movieid'}, inplace=True)
    datafout['userid'] = datafout['userid'].astype(float)#change the type from str to float
    datafout['movieid'] = datafout['movieid'].astype(float)
    datafout['rating'] = dataf['Prediction']
    datafout['rating'] = datafout['rating'].astype(float)
    return datafout
def format_data(data):
	"""format the data as a dictionary"""
	ratings = {}
	for userId, movieId, rating in data:
		if not userId in ratings: 
			ratings[userId] = {}
		ratings[userId][movieId] = rating

	return ratings

def create_dataset(path_methods,path_true_val,filename):
	"""take multiple submissions and merge them into one joint dataset"""

	methods = {}
	methods_arr = []

	for f in os.listdir(path_methods):
		methods[f] = format_data(seperate_data(path_methods + str(f)))
		methods_arr.append(f)
		
	true_values = format_data(seperate_data(path_true_val))

	result_f = open(filename,"w")

	result_f.write("Id,Prediction")

	for method in methods_arr:
		result_f.write("," + str(method))

	result_f.write("\n")

	first = next (iter (methods.values()))

	for r in first:
		for c in first[r]:
			result_f.write("r" + str(r) + "_c" + str(c) + ",")
			result_f.write(str(true_values[r][c]))
			
			for method in methods_arr:
				pred = methods[method][r][c]
				row = "," + str(pred)
				result_f.write(row)
			result_f.write("\n")

	result_f.close()
    

def create_submission(data, filename="submission.csv"):
	print("Creating submission " + str(filename))
	f = open(filename,"w")
	f.write("Id,Prediction\n")

	for user in data:
		for movie in data[user]:
			rating = data[user][movie]
			f.write('r{0}_c{1},{2}'.format(user,movie,rating) + "\n")
	f.close()
    
def create_input(user_ID, movie_ID, ratings):
    """

    :param user_id: a list containing the id of users
    :param movie_id: a list containing the id of movies
    :param ratings: a list containing the corresponding ratings
    :return: Interaction Object (a useful object containing users, movies and ratings)
    """
    return Interactions(user_ID, movie_ID, ratings)
def create_model(loss, k, number_epochs, batch_size, l2_penal, gamma):
    """

    :param loss: The loss we want to use for our optimization process
    :param k: the latent dimension of our matrix factorization
    :param number_epochs: the number of times we want to go through all our training set during the training phase
    :param batch_size: the size of the batch to perform optimization algorithm
    :param l2_penal: ridge penalization (L2)
    :param gamma: our optimization learning rate
    :return: a factorization model ready to fit our input data
    """
    model = ExplicitFactorizationModel(loss=loss,
                                       embedding_dim=k,  # latent dimensionality
                                       n_iter=number_epochs,  # number of epochs of training
                                       batch_size=batch_size,  # minibatch size
                                       l2=l2_penal,  # strength of L2 regularization
                                       learning_rate=gamma)
    return model
def train_model(model, X):
    """

    :param model: Our model
    :param X: the data on which we wish to train our model
    :return: our model trained
    """
    model.fit(X, verbose=True)
    return model
def predict_output(model, test_interactions_object):
    """

    :param model: our trained model
    :param test_interactions_object: the interaction object on which we wish to predict ratings
    :return: desired output
    """
    y_predictions = model.predict(test_interactions_object.user_ids, test_interactions_object.item_ids)
    return np.round(y_predictions)
def create_output_df(y_predictions, test_df):
    """

    :param y_predictions: the predictions our model gave
    :param test_df: our test dataframe in the submission form
    :return: pandas dataframe with our predictions
    """
    test_df['Prediction'] = y_predictions
    test_df['Prediction'] = test_df['Prediction'].apply(lambda x: 1 if x < 1 else x)
    test_df['Prediction'] = test_df['Prediction'].apply(lambda x: 5 if x > 5 else x)
    return test_df

def create_submission_pd(test_df):
    """

    :param test_df: our final submission in pandas dataframe format
    :return: csv of our final submission
    """
    test_df.to_csv('pytorch_submission.csv', index=False)
    
def parse_line(line):
    """Extract row and column numbers as well as the rating from a line."""
    rrow_ccol, rating = line.split(",")
    rrow, ccol = rrow_ccol.split("_")
    return int(rrow[1:]), int(ccol[1:]), rating


def get_submission_csv(item_features, user_features, path_sample, path_submission):
    """Creates a csv submission file in the same format as the sample submission using the predicted user and item features."""
    predicted_ratings = item_features.T @ user_features
    print(predicted_ratings.shape)
    line_counter = 0
    with open(path_sample, mode='r') as sample:
        with open(path_submission, mode='w') as submission:
            for line in sample:
                if line_counter == 0:
                    submission.write(line)
                    line_counter += 1
                    continue
                row, col, rating = parse_line(line)
                rating = int(round(predicted_ratings[row-1, col-1]))
                if rating not in [1,2,3,4,5]:
                    if rating > 5:
                        rating = 5
                    if rating < 1:
                        rating = 1
                if rating not in [1,2,3,4,5]:
                    print("Error: submission format is not respected!")
                new_line = "r" + str(row) + "_" + "c" + str(col) + "," + str(rating) + "\n"
                submission.write(new_line)
                line_counter += 1
    print("Submission created successfully!")