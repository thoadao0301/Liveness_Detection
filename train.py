import os
from keras import layers
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import  ModelCheckpoint, LearningRateScheduler, TensorBoard
from datetime import datetime

from Siamese_network import SiameseNetwort as S
import argparse

def scheduler(epoch,lr):
    if epoch < 70:
        return lr
    elif epoch < 100:
        return lr/1.01
    else:
        return lr/1.001
def main(args):
    DATA_PATH = args.numpy_data_directory
    X_data = 'X_data.npy'
    y_data = 'y_data.npy'

    MODEL_PATH = args.model_checkpoint_path
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    CHKPT_FILE_NAME = subdir + '.ckpt'
    MODEL_FILE_NAME = 'siamese-liveness-model.h5'


    X = np.load(os.path.join(DATA_PATH,X_data))
    X = np.array(X,dtype='float')/255.0
    y = np.load(os.path.join(DATA_PATH,y_data))

    print('Number of pair images: ', X.shape[0])

    X_train, X_test, y_train,  y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    input_dim = X_train.shape[2:]

    img_a = Input(shape=input_dim)
    img_b = Input(shape=input_dim)


    base_network = S.model_with_base_inception_resnet_v1(input_dim)
    base_network.summary()

    feat_vecs_a = base_network(img_a)
    feat_vecs_b = base_network(img_b)

    print(feat_vecs_a)
    print(feat_vecs_b)

    distance = Lambda(S.euclidean_distance)([feat_vecs_a,feat_vecs_b])

    outputs = layers.Dense(1,activation='sigmoid')(distance)

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    MODEL_CHECKPOINT_PATH = os.path.join(MODEL_PATH,CHKPT_FILE_NAME)
    print('Check point file path: ', MODEL_CHECKPOINT_PATH)

    checkpointer = ModelCheckpoint(MODEL_CHECKPOINT_PATH, verbose=1, save_best_only=True)

    lr_decrease = LearningRateScheduler(scheduler)

    if args.model_log_path is not None:
        tensorboard_callback = TensorBoard(log_dir=args.model_logs_path, histogram_freq=1)
        callbacks=[checkpointer,lr_decrease,tensorboard_callback]
    else:
        callbacks=[checkpointer,lr_decrease]
    
    model = Model(inputs=[img_a,img_b],outputs=outputs)

    optimizer = Adam(args.learning_rate)
    model.compile(loss=S.constrastive_loss, optimizer=optimizer,metrics=['accuracy'])

    img_1 = X_train[:,0]
    img_2 = X_train[:,1]

    print('-----Training-----')

    model.fit([img_1,img_2],y_train,\
        validation_data=(X_test[:,0],X_test[:,1],y_test),\
        batch_size=args.batch_size,verbose=2,epochs=args.epoch,callbacks=callbacks)

    model.save(os.path.join(MODEL_PATH,MODEL_FILE_NAME))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('numpy_data_directory', type=str,
                        help='Path to the dataset by numpy file')

    parser.add_argument('--model_checkpoint_path', type=str,
                        help='Path to the checkpoint directory', default=None)

    parser.add_argument('--model_logs_path', type=str,
                        help='Path to the logs directory', default=None)

    parser.add_argument('--learning_rate', type=float,
                        help='Default is 0.001', default=0.001)

    parser.add_argument('--epochs', type=int,
                        help='Default is 90', default=90)

    parser.add_argument('--batch_size', type=int,
                        help='Default is 32', default=32)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
