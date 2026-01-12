import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py as h5
import sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_recall_curve, accuracy_score, confusion_matrix  
from sklearn.metrics import average_precision_score  
import pickle
import keras
from keras.models import load_model
from pathlib import Path
from keras.optimizers import Adam
from scipy.io import loadmat
import copy
from datetime import datetime
import json
import argparse
import tensorflow as tf
from keras.callbacks import TensorBoard
import pickle as pk



def parser_args():
    parser = argparse.ArgumentParser()

    # repo and model path
    parser.add_argument("--db_h5_file", default="./deepbeat.h5")
    # data path
    parser.add_argument("--orig_data_path", default= r'C:\Users\aoara\develop\deepbeat\data\original_data')
    parser.add_argument("--relabled_path", default=r'C:\Users\aoara\develop\deepbeat\data\relabeled_data')  
    
    # output path
    parser.add_argument("--output_path", default= r'C:\Users\aoara\develop\deepbeat\training_output')

    # experiment config
    parser.add_argument("--file_name", required= True, help="name the file (model name)")
    valid_choices = ['db_orig', 'db_relabel', 'db_relabel_w_vsm', 'db_orig_replaced', 'db_orig_replaced_vsm']
    # db_relabel means remove all old data, keep relabel data ONLY
    # db_orig_replaced means substitute old data with relabeled data, keep non-relabeled data as they are
    parser.add_argument("--training_choice", choices= valid_choices, required= True, help="training data choice: " + str(valid_choices))
    parser.add_argument("--db_orig_replaced_path", default= r"C:\Users\aoara\develop\deepbeat\output\replace_relabeled.pkl")
    
    # hyperparameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    

    #optimizer
    # parser.add_argument("--AdamW", action="store true", help= "use AdamW instead of default Adam optimizer")
    # parser.add_argument("--weight_decay", type=float, default=5e-4, help= "weight decay with AdamW")

    # specify type of relabeled data

    
    
    args = parser.parse_args()

    return args

def setup_tensorboard(args):
    log_path = Path(args.output_path) / Path(args.file_name)
    log_path.mkdir(parents=True, exist_ok=True)
    
    tensorboard_callback = TensorBoard(
        log_dir=str(log_path),
        histogram_freq=1,  # Log weight histograms every epoch
        write_graph=True,  # Visualize the graph
        write_images=False,
        update_freq='epoch',  # Log metrics every epoch
        profile_batch=0,  # Disable profiling
        embeddings_freq=0
    )

    return tensorboard_callback

    
    
def get_orig_deepbeat(args):
    with h5.File(args.db_h5_file, 'r') as f:    
        training_config = json.loads(f.attrs['training_config'])
        
    orig_config = training_config ['optimizer_config']['config']
    orig_config['learning_rate'] = orig_config.pop('lr', None) # rename lr to learning rate
    orig_config.pop('decay', None) # there is no longer a parameter called decay; the original decay was 0

    # load deepbeat model with new tensorflow package, verify performances
    deepbeat = load_model( args.db_h5_file, compile = False) 
    
    return deepbeat, orig_config


def remove_nan_data(data_dict):
    # get non-nan signals
    no_nan_mask = ~np.isnan(data_dict['data']).any(axis=(1, 2))
    
    for k in data_dict.keys():
        data_dict[k] = data_dict[k][no_nan_mask]
    
    return data_dict

def load_original_data(data_path, file_name):
    data = np.load(Path(data_path) / file_name,allow_pickle=True )
    output = {}
    output['data'] = data['signal']
    output['qa_label'] = data['qa_label']
    output['rhythm'] = data['rhythm'] 
    params = pd.DataFrame(data['parameters'])
    params.rename(index=str, columns={0:'timestamp', 
                                  1:'stream', 
                                  2:'ID'}, inplace=True)                            
    output['ID'] = np.array(params['ID'].to_list())
    
    # a portion of the original data contains nan (for signal that contains nan, all data in signal are nan)
    # remove these signals
    output = remove_nan_data(output)
    
    return output



def load_relabeled_data(data_path):
    # return combinbed, relabeled_db, relabeled_VSM
    #['data'], ['qa_label'], ['rhythm'], ['parameters'], ['ID']

    def load_from_mat(dir_path, file_name):
        file_mat = loadmat(Path(dir_path) / file_name)
        file = file_mat.get(file_name[:-4])
        return file 
    combined = {}
    combined['data'] = load_from_mat(data_path,'db_vsm_combined_data.mat' )
    combined['qa_label'] = load_from_mat(data_path, 'db_vsm_combined_label_q.mat' )
    combined['rhythm'] = load_from_mat(data_path, 'db_vsm_combined_label_r.mat' )
    combined['ID'] =load_from_mat(data_path, 'db_vsm_combined_sub_id.mat').flatten()
    # reshaping to original data
    # reshaping to match db's original data
    combined['data'] = combined['data'].reshape(combined['data'].shape[0], combined['data'].shape[1], 1)
    num_classes_rhythm = 2
    num_classes_qa = 3
    # one-hot encoding
    combined['rhythm']= keras.utils.to_categorical(combined['rhythm'], num_classes_rhythm)
    combined['qa_label'] = keras.utils.to_categorical(combined['qa_label'], num_classes_qa)
    
    relabeled_db = {}
    relabeled_vsm = {}
    
    # VSM index starts from 1000
    db_mask = (combined['ID'] < 1000).flatten()
    vsm_mask = (combined['ID']>=1000).flatten()
    # separate the db data
    relabeled_db['data'] = combined['data'][db_mask,:]
    relabeled_db['qa_label'] = combined['qa_label'][db_mask, :]
    relabeled_db['rhythm'] = combined['rhythm'][db_mask, :]
    relabeled_db['ID'] =  combined['ID'][db_mask].flatten()
    # separate the vsm data
    relabeled_vsm['data'] = combined['data'][vsm_mask, :]
    relabeled_vsm['qa_label'] = combined['qa_label'][vsm_mask, :]
    relabeled_vsm['rhythm'] = combined['rhythm'][vsm_mask,:]
    relabeled_vsm['ID'] = combined['ID'][vsm_mask].flatten()
    
    return combined, relabeled_db , relabeled_vsm


def replace_updated_subjects_db(db_train, relabeled_db):
    """
    for each relabeled subject, remove old data, and replace it with new data
    """
    
    subjects_to_replace = np.unique(relabeled_db['ID'])
    mask_keep = ~np.isin(db_train['ID'], subjects_to_replace)
    
    db_train['data'] = db_train['data'][mask_keep]
    db_train['rhythm'] = db_train['rhythm'][mask_keep]
    db_train['qa_label'] = db_train['qa_label'][mask_keep]
    db_train['ID'] = db_train['ID'][mask_keep]
    
    db_train['data'] = np.concatenate([db_train['data'], relabeled_db['data']], axis=0)
    db_train['rhythm'] = np.concatenate([db_train['rhythm'], relabeled_db['rhythm']], axis=0)
    db_train['qa_label'] = np.concatenate([db_train['qa_label'], relabeled_db['qa_label']], axis=0)
    db_train['ID'] = np.concatenate([db_train['ID'], relabeled_db['ID']], axis=0)
     
    return db_train

def load_substituted_relabeled_data(path):

    """
    load the saved substituted original data (keep un-relabel data)
    """
    with open(path, 'rb') as file:
        orig_sub_relabel = pk.load(file)
    
    return orig_sub_relabel

def attach_VSM (db_data, relabeled_vsm):
    db_data['data'] = np.concatenate([db_data['data'], relabeled_vsm['data']], axis=0)
    db_data['rhythm'] = np.concatenate([db_data['rhythm'], relabeled_vsm['rhythm']], axis=0)
    db_data['qa_label'] = np.concatenate([db_data['qa_label'], relabeled_vsm['qa_label']], axis=0)
    db_data['ID'] = np.concatenate([db_data['ID'], relabeled_vsm['ID']], axis=0)
    return db_data

def shuffle_data(db_train):
    """

    Args:
        db_train (dict): keys - 'data', 'qa_label', 'rhythm', 'ID'
    """
    data_train, label_train_r, label_train_q = db_train['data'], db_train['rhythm'], db_train['qa_label']
    # random shuffle
    idx = np.random.permutation(range(len(label_train_r)))  # shuffled indices
    # shuffle together
    data_train, label_train_r, label_train_q = data_train[idx, :], label_train_r[idx], label_train_q[idx]
    
    return data_train, label_train_r, label_train_q

def main():
    # check gpu status

    print("TENSORFLOW GPU STATUS")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"  - {gpu}")
    print("="*60 + "\n")

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    # seed everything
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # parse ags
    args = parser_args()

    
    # load model
    db_trained, orig_config = get_orig_deepbeat(args)
    
    # clone model  (new, does not preserve old weights)
    new_db = keras.models.clone_model(db_trained)

    # optimizer config
    # if args.AdamW:
    #     optimizer = AdamW(weight_decay= args.weight_decay)
    # else:
    #     optimizer = Adam( **orig_config)

    optimizer = Adam( **orig_config)
    new_db.compile(
        optimizer= optimizer,
        loss={
            'qa_output': 'categorical_crossentropy',
            'rhythm_output': 'binary_crossentropy' 
        },
        loss_weights={
            'qa_output': 0.2,      
            'rhythm_output': 5.0   
        },
        metrics={'rhythm_output': 'accuracy', 'qa_output': 'accuracy'})

    # Samiya's loss
    # .compile(optimizer=tf.keras.optimizers.Adam(),
    #           loss={'rhythm_output': BinaryFocalLoss(gamma=2), 'qa_output': 'categorical_crossentropy'},
    #           loss_weights={'rhythm_output': 1, 'qa_output': 1},
    #           metrics={'rhythm_output': 'accuracy', 'qa_output': 'accuracy'})
    
    #prepare training data
    # load data
    print("loading training data")
    
    def load_training_data(args):
        print("=" * 60)
        print(f"TRAINING CHOICE: {args.training_choice}")
        print("=" * 60)
        
        # db_orig_replaced: replace relabeled data, keep unrelabeled data
        if args.training_choice in ["db_orig_replaced", "db_orig_replaced_w_vsm"]:
            data_to_shuffle = load_substituted_relabeled_data(args.db_orig_replaced_path)
            
            if args.training_choice == "db_orig_replaced_w_vsm":
                _, _, relabeled_vsm = load_relabeled_data(args.relabled_path)
                return attach_VSM(data_to_shuffle, relabeled_vsm)
            
            return data_to_shuffle
        
        # Handle db_orig
        if args.training_choice == "db_orig":
            return load_original_data(args.orig_data_path, 'train.npz')
        
        # db_relabel: keep ONLY relabeled data
        if args.training_choice in ["db_relabel", "db_relabel_w_vsm"]:
            db_train = load_original_data(args.orig_data_path, 'train.npz')
            _, relabeled_db, relabeled_vsm = load_relabeled_data(args.relabled_path)
            data_to_shuffle = replace_updated_subjects_db(db_train, relabeled_db)
            
            if args.training_choice == "db_relabel_w_vsm":
                return attach_VSM(data_to_shuffle, relabeled_vsm)
        
        return data_to_shuffle

    data_to_shuffle = load_training_data(args)

    data_train, label_train_r, label_train_q = shuffle_data(data_to_shuffle)
    
    
    # perpare validation data
    db_val = load_original_data(args.orig_data_path, 'validate.npz')
    data_val, label_val_r, label_val_q = db_val['data'], db_val['rhythm'], db_val['qa_label']
    
    print("training starts")

    tensorboard_callback = setup_tensorboard(args)
    history = new_db.fit(
        data_train, 
        {"rhythm_output": label_train_r, "qa_output": label_train_q},
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(data_val, {"rhythm_output": label_val_r, "qa_output": label_val_q}),
        callbacks=[tensorboard_callback],
        verbose=1
    )
    # save model and history
    print("saving trained model and history")
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_dir = output_path / Path(args.file_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    new_db.save( model_dir  / (args.file_name + '.keras'))
    
    all_history = {'model_name': args.file_name+'.keras',
                   'training_data': args.training_choice,
                   'date': datetime.now().isoformat(),
                   'history': history.history
                   }
 
    with open( model_dir /(args.file_name + '_history.pkl'), 'wb') as file:
        pickle.dump(all_history, file)
        

if __name__ == "__main__":
    main()