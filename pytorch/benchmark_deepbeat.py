import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_recall_curve, accuracy_score, confusion_matrix, average_precision_score
import os
def episode_metrics_equiv(preds_df, out_message=False):
    """
    Args:
    preds_df: pd.Datafram containing all
    - rhythm predictions
    - QA predictions
    - QA ground truth
    - rhythm ground truth
    - IDs
    for all individuals
    """
    y_predictions= preds_df['rh_pred']
    y_truth = preds_df['rh_true']
    # Confusion_matrix
    cf = confusion_matrix(y_truth, y_predictions, labels=[0, 1])
    TN, FP, FN, TP = cf.ravel()
    support = TN+FP+FN+TP
    # Sensitivity, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # F1 score
    f1 = metrics.f1_score(y_truth, y_predictions, average=None)
    # selecting f1score for positive case if present
    f1_pos = f1[1] if len(f1) > 1 else f1[0]
    # Area under precision recall curve
    #auprc = metrics.average_precision_score(y_truth, rhythm[:,1])
    if out_message:
        print(pd.DataFrame(cf).rename(columns={0: "Predicted Non-AF", 1:" Predicted AF"}, index={0: "True Non-AF", 1:"True AF"}))
        print("Sensitivity/Recall: %0.4f" % TPR)
        print("Specificity: %0.4f" % TNR)
        print("Precision/PPV: %0.4f" % PPV)
        print("Negative predictive value/NPV: %0.2f" % NPV)
        print("False positive rate: %0.4f" % FPR)
        print("False negative rate: %0.4f" % FNR)
        print("F1 score: %0.4f" % f1_pos)
        print('support: ', support)
        print('\n\n\n')
    episode_metrics = [TPR, TNR, PPV, NPV, FPR, FNR, f1_pos, support]
    return episode_metrics


# Index(['rh_true', 'rh_pred', 'qa_true', 'qa_pred', 'ID'], dtype='object')
def collecting_individual_metrics_equiv( preds_df, out_message=True):
    """
    Args:
    preds_df: pd.Datafram containing all
    - rhythm predictions
    - QA predictions
    - QA ground truth
    - rhythm ground truth
    - IDs
    for all individuals
    """
    predictions_r = preds_df['rh_pred']
    rhythm = preds_df['rh_true']
    IDs = preds_df['ID']

    individual_metrics = {}
    for i in np.unique(IDs):
        # Sub-selecting individuals
        p_indx = (IDs == i)
        # filter predictions and ground truth
        pred_r_filtered = predictions_r[p_indx]
        rhythm_filtered = rhythm[p_indx]

        # Confusion_matrix
        cf = confusion_matrix(rhythm_filtered,pred_r_filtered, labels=[0, 1])
        TN, FP, FN, TP = cf.ravel()
        support = TN+FP+FN+TP
        # Sensitivity, recall, or true positive rate
        TPR = TP/(TP+FN) if (TP+FN) > 0.0 else np.nan
        # Specificity or true negative rate
        TNR = TN/(TN+FP) if (TN+FP) > 0.0 else np.nan

        # Fall out or false positive rate
        FPR = FP/(FP+TN) if (FP+TN) > 0.0 else np.nan
        # False negative rate
        FNR = FN/(TP+FN) if (TP+FN) > 0.0 else np.nan
        # F1 score
        f1 = metrics.f1_score(rhythm_filtered, pred_r_filtered , average=None)
        # selecting f1score for positive case if present
        f1_pos = f1[1] if len(f1) > 1 else f1[0]
        # Area under precision recall curve
        #auprc = metrics.average_precision_score(y_test_truth, rhythm_pID[:,1])
        individual_metrics[i] = [TPR, TNR, FPR,FNR, f1_pos, support ]

        if out_message:
            print("PATIENT :", i , '\n')
            print(pd.DataFrame(cf).rename(columns={0: "Predicted Non-AF", 1:" Predicted AF"}, index={0: "True Non-AF", 1:"True AF"}))
            if (TP + FN) > 0:
                print("Sensitivity/Recall: %0.4f" % TPR)
                print("False negative rate: %0.4f" % FNR)
                print("F1 score: %0.4f" % f1_pos)
                print('support: ' , support)
            if (FP + TN) > 0:
                print("Specificity: %0.4f" % TNR)
                print("False positive rate: %0.4f" % FPR)
                print('support: ' , support)
    return individual_metrics



def get_quality_signals(preds_df, level = 2):
  """
  filter down data based on PREDICTED signal quality
  """
  # Index(['rh_true', 'rh_pred', 'qa_true', 'qa_pred', 'ID'], dtype='object')
  output = {}

  # excellent = 2
  excellent_qa_indx = np.where(preds_df['qa_pred']==level)[0]
  for key in preds_df.keys():
    output[key] = preds_df[key][excellent_qa_indx]

  return output



def deepbeat_metrics(preds_df, level):
  """
  Args:
   preds_df: pd.Datafram containing all
    - rhythm predictions
    - QA predictions
    - QA ground truth
    - rhythm ground truth
    - IDs
    for all individuals

   level: signal quality level

  deepbeat paper
  first filter signal based on PREDICTED signal quality, then
  calculates 'TPR', 'TNR', 'FPR', 'FNR' at PATIENT level
  calculates 'PPV', 'NPV', 'F1' at EPISODE level
  """
  output = {}
  filtered = get_quality_signals(preds_df, level)
  print('QUALIY LEVEL: '+ str(level))
  test_metrics_1 = collecting_individual_metrics_equiv(filtered, out_message=False)
  test_metrics = pd.DataFrame.from_dict(test_metrics_1).T.rename(columns={0:'TPR', 1:'TNR', 2:'FPR', 3:'FNR', 4: 'F1', 5:"total_samples"})


  for m in ['TPR', 'TNR', 'FPR', 'FNR']:
      metric_wmu = np.average(test_metrics[m][~test_metrics[m].isna()], weights=test_metrics['total_samples'][~test_metrics[m].isna()])
      print('%s: %0.2f' % (m, metric_wmu))
      output[m] =  metric_wmu
  # PPV, NPV and F1
  episode_m = episode_metrics_equiv(filtered, out_message=False)
  episode_metrics = pd.DataFrame(episode_m).T
  episode_metrics.rename(columns={0:'TPR', 1:'TNR', 2:'PPV', 3:'NPV', 4:"FPR", 5:'FNR', 6:'F1', 7:'total_samples'}, inplace=True)
  for m in ['PPV', 'NPV', 'F1']:
      print('%s: %0.2f' % (m, episode_metrics[m]))
      output[m] = episode_metrics[m].squeeze()
  return output


def organize_results(prediction_path, path_to_test_data):
  data_test = np.load(path_to_test_data, allow_pickle=True)
  preds = pd.read_csv(prediction_path)
  preds['ID'] = data_test['ID']
  # for rhythm branch only model
  if "qa_pred" not in preds.columns:
    preds['qa_pred'] = data_test['qa_label']
  return preds




def output_metrics(output0_dict, output1_dict, output2_dict, study_name):
  df0 = pd.DataFrame([output0_dict])
  df1 = pd.DataFrame([output1_dict])
  df2 = pd.DataFrame([output2_dict])

  df0['qa level'] = 0
  df1['qa level'] = 1
  df2['qa level'] = 2

  new_df = pd.concat([df0, df1, df2], ignore_index=True)
  new_df['study'] = study_name

  file_path = "/content/drive/MyDrive/afib/benchmark_metrics.csv"

  if os.path.exists(file_path):
    existing_df = pd.read_csv(file_path)
    if 'Unnamed: 0' in existing_df.columns:
        existing_df = existing_df.drop(columns=['Unnamed: 0'])
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(file_path, index=False)
    print('append to csv!')
  else:
    new_df.to_csv(file_path, index=False)
    print('create new csv!')
