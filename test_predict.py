import os
import numpy as np
from keras.models import load_model
from Siamese_network import SiameseNetwort as S

def main(args):
    DATA_TEST_PATH = args.test_dataset_npy
    X_data = 'X_data.npy'
    y_data = 'y_data.npy'
    MODEL_PATH = args.model_path
    model = load_model(MODEL_PATH,custom_objects={'contrastive_loss' : S.constrastive_loss})

