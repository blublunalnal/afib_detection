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
sys.path.append(r'C:\Users\aoara\OneDrive\Documents\repos\deepbeat')
import utils
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from scipy.io import loadmat
import copy
import pickle  
from datetime import datetime
import json


def get_orig_deepbeat():
    with h5.File(r"C:\Users\aoara\OneDrive\Documents\repos\deepbeat\deepbeat.h5", 'r') as f:
    # Check for version attributes
        # if 'keras_version' in f.attrs:
        #     print(f"Keras version: {f.attrs['keras_version']}")
        
        # if 'backend' in f.attrs:
        #     print(f"Backend: {f.attrs['backend']}")
        
        # # Sometimes stored under model config
        # if 'model_config' in f.attrs:
        #     import json
        #     config = json.loads(f.attrs['model_config'])
        #     if 'keras_version' in config:
        #         print(f"Keras version (from config): {config['keras_version']}")
    
        training_config = json.loads(f.attrs['training_config'])
        
    orig_config = training_config ['optimizer_config']['config']
    orig_config['learning_rate'] = orig_config.pop('lr') # rename lr to learning rate
    orig_config.pop('decay') # there is no longer a parameter called decay; the original decay was 0

    # load deepbeat model with new tensorflow package, verify performances
    path_to_model =r'C:\Users\aoara\OneDrive\Documents\repos\deepbeat'
    model_name = 'deepbeat.h5'
    deepbeat = load_model( Path(path_to_model) / model_name, compile = False) 
    
    return deepbeat, orig_config


def remove_nan_data( data_dict):
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

def load_from_mat(dir_path, file_name):
    file_mat = loadmat(Path(dir_path) / file_name)
    file = file_mat.get(file_name[:-4])
    return file 

def load_relabeled_data(data_path):
    # return combinbed, relabeled_db, relabeled_VSM
    #['data'], ['qa_label'], ['rhythm'], ['parameters'], ['ID']
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
    vsm_mask = (combined['ID']> 1000).flatten()
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
    
    file_name = input("name the file (model name): ").strip().lower()
    
    training_choice = input("training data choice - db_orig, db_relabel, db_relabel_w_vsm").strip().lower()
    
    valid_choices = ['db_orig', 'db_relabel', 'db_relabel_w_vsm']
    if training_choice not in valid_choices:
        print(f"invalid choice. select from: {', '.join(valid_choices)}")
        return
    
    
    # load data
    print("loading data")
    orig_data_path = Path(r'C:\Users\aoara\OneDrive\Documents\repos\deepbeat\data')
    #db_test = load_original_data(orig_data_path, 'test.npz')
    db_train = load_original_data(orig_data_path, 'train.npz')
    relabled_path = Path(r'C:\Users\aoara\OneDrive\Documents\repos\deepbeat\samiya\Model Development (AFib Detection)\Labelled Data\combined data (VSM+DeepBeat relabelled)')
    relabeled_combined, relabeled_db, relabeled_vsm = load_relabeled_data(relabled_path)
    # merged old data with relabeled data
    db_train_copy = copy.deepcopy(db_train)
    db_train_update = replace_updated_subjects_db(db_train_copy, relabeled_db)
    db_VSM_train = attach_VSM(db_train_update, relabeled_vsm)
    print("finished processing data")
    # load model
    db_trained, orig_config = get_orig_deepbeat()
    
    # clone model  (new, does not preserve old weights)
    new_db = keras.models.clone_model(db_trained)
    new_db.compile(
        optimizer= Adam( **orig_config),
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
    
    training_type_dict = {'db_orig': db_train, 
                          'db_relabel': db_train_update,
                          'db_relabel_w_vsm': db_VSM_train}
    
    data_to_shuffle = training_type_dict[training_choice]
    
    print("training starts")
    
    
    data_train, label_train_r, label_train_q = shuffle_data(data_to_shuffle)
    
    batch_size = 128
    epochs = 100
    # Train Model
    history = new_db.fit(data_train, {"rhythm_output": label_train_r, "qa_output": label_train_q},
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)
    # save model and history
    print("saving trained model and history")
    output_path = Path(r'C:\Users\aoara\OneDrive\Documents\repos\deepbeat\new_models')
    output_path.mkdir(parents=True, exist_ok=True)
    new_db.save(output_path / (file_name + '.keras'))
    
    all_history = {'model_name': file_name+'.keras',
                   'training_data': training_choice,
                   'date': datetime.now(),
                   'history': history.history
                   }
 
    with open(output_path / (file_name + '_history.pkl'), 'wb') as file:
        pickle.dump(all_history, file)
        

if __name__ == "__main__":
    main()