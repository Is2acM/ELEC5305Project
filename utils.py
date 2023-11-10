'''

this script is for the helper functions code of Project 2..

-------------------------------------------
'''

import torch
import matplotlib
import matplotlib.pyplot as plt
import os
import logging
import argparse
import time
import sys
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd 
import numpy as np
matplotlib.style.use('ggplot')

outputs_path = './outputs'
if not os.path.exists(outputs_path):
    os.makedirs(outputs_path)

# Save the Trained Model for future evaluations
def save_model(model):
    """Function to save the trained model."""
    torch.save(model.state_dict(), os.path.join(outputs_path,'model.pth'))

# Save the loss and accuracy performance plots
def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """Function to save the loss and accuracy plots while training."""
    epochs = range(1,len(train_acc)+1)

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        epochs,train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        epochs,valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(outputs_path,'accuracy.png'))
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        epochs,train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        epochs,valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(outputs_path,'loss.png'))

# Save Confusion Matrix
def saveCM(y_true, y_pred,classes):
    ''' Plot Confusion Matrix'''
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                    columns = [i for i in classes])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(outputs_path,'CM.png'))

# control input options. DO NOT CHANGE THIS PART.
def parse_args():
    '''Function to read user inputs. '''
    parser = argparse.ArgumentParser(description='Main scipt for ELEC5305 Research Project')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Used when there are cuda installed.')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs to train the network')
    pargs = parser.parse_args()
    return pargs

# Creat logs. DO NOT CHANGE THIS PART.
def create_logger(final_output_path = outputs_path):
    '''Function to create log files '''
    
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger

