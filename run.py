import pandas as pd
import torch
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score
from helpers import *
torch.set_default_tensor_type('torch.DoubleTensor')
LOSS = 'regression'  # Our chosen loss
K = 20  # Latent dimension of our matrix factorization
NB_EPOCHS = 30  # Number of times we go through our training set
BATCH_SIZE = 32  # The batch size of our optimization algorithm
L2 = 1e-5  # Our lambda ridge penalization
GAMMA = 1e-4  # Our optimization learning rate

# Loading train data
print("---------LOADING DATA-----------")
raw_data_train = pd.read_csv('data_train.csv')
raw_data_output = pd.read_csv('sampleSubmission.csv')

print("---------PREPROCESSING DATA-----------")
df_input = split_(raw_data_train, column='Id')
df_output = split_(raw_data_output, column='Id')
input_interaction = create_input(df_input['userid'].values,
                                               df_input['movieid'].values,
                                               df_input['rating'].values)

output_interaction = create_input(df_output['userid'].values,
                                                df_output['movieid'].values,
                                                df_output['rating'].values)

print("---------TRAINING THE MODEL-----------")
model = create_model(loss=LOSS, k=K, number_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, l2_penal=L2, gamma=GAMMA)
model = train_model(model, input_interaction)
print("---------CREATING THE SUBMISSION-----------")
df_submission = create_output_df(y_predictions=predict_output(model, output_interaction), test_df=raw_data_output)
create_submission_pd(df_submission)



