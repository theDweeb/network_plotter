'''
    Takes the output CSV file (if more than one it will merge them together)
    and plots the top1/5 accuracy as well as the validation and training loss
'''
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob

parser = argparse.ArgumentParser(description="Network Plotter")
parser.add_argument('--csv-path', dest='csv_path', default='csv', type=str,
                    help='path to csv(s)')
parser.add_argument('--filename', default="summary.csv", type=str,
                    help="Filename of the complete CSV")

args = parser.parse_args()
CWD = os.getcwd()
CSV = args.csv_path

def _concat_csv(all_filenames):
    print("Merging CSV files...")
    # Combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

    # Delete individual CSVs
    for f in all_filenames:
        os.remove(f)

    # Re-sort by epoch
    combined_csv = combined_csv.sort_values((["epoch"]), ascending="False")
    combined_csv = combined_csv.reset_index(drop=True)

    # Export to CSV
    print("CSV's merged into 'summary.csv'")
    combined_csv.to_csv("summary.csv", index=False, encoding='utf-8-sig')
    
    # Return merged CSV
    return combined_csv

def _get_csv():    
    os.chdir(CSV)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    if len(all_filenames) > 1:
        summary = _concat_csv(all_filenames)
        os.chdir(CWD)
        return summary
    else:
        summary = pd.read_csv(args.filename)
        os.chdir(CWD)
        return summary

def _plot_error(summary):
    x = summary[['epoch']]
    loss = summary[['train_loss', 'eval_loss']]

    train = loss['train_loss'].values
    val = loss['eval_loss'].values

    _, (p1, p2, p3) = plt.subplots(3, 1, figsize=(12,7))

    p1.set_title("Training Loss vs Validation Loss")
    p1.plot(x, loss['train_loss'], label="Training loss")
    p1.plot(x, loss['eval_loss'], label="Validation loss")
    for var in (train, val):
        p1.annotate('%0.2f' % var.min(), xy=(1, var.min()), xytext=(8, 0), 
            xycoords=('axes fraction', 'data'), textcoords='offset points')
    p1.set_xlabel("Epochs")
    p1.set_ylabel("Error")
    p1.legend(loc='upper right')

    p2.set_title("Training Loss: Last 20 Epochs")
    p2.plot(x.tail(20), loss['train_loss'].tail(20), label="Training loss")
    p1.set_xlabel("Epochs")
    p1.set_ylabel("Error")
    p2.legend(loc='upper right')

    p3.set_title("Validation Loss: Last 20 Epochs")
    p3.plot(x.tail(20), loss['eval_loss'].tail(20),'orange', label="Validation loss")
    p1.set_xlabel("Epochs")
    p1.set_ylabel("Error")
    p3.legend(loc='upper right')

    _.subplots_adjust(hspace=1)
    plt.savefig("Training Loss", bbox_inches = 'tight')

def _plot_accuracy(summary):
    x = summary[['epoch']]
    prec = summary[['eval_prec1', 'eval_prec5']]

    top1 = prec['eval_prec1'].values
    top5 = prec['eval_prec5'].values

    _, (p1, p2, p3) = plt.subplots(3, 1, figsize=(12,7))

    p1.set_title('Top 1 vs Top 5 Validation Accuracy')
    p1.plot(x, prec['eval_prec1'], 'r', label="Top 1")
    p1.plot(x, prec['eval_prec5'], 'b', label="Top 5")
    for var in (top1, top5):
        p1.annotate('%0.2f' % var.max(), xy=(1, var.max()), xytext=(8, 0), 
            xycoords=('axes fraction', 'data'), textcoords='offset points')
    p1.set_xlabel("Epochs")
    p1.set_ylabel("Validation Accuracy")
    p1.legend(loc='upper right')

    max_acc = np.max(prec['eval_prec1'].values)
    max_indx = np.argmax(prec['eval_prec1'].values)

    print("Highest top-1 accuracy: %0.2f @ epoch: [%i/%i]" % (max_acc, max_indx,x.size))

    p2.set_title("Top 1: Last 20 epochs")
    p2.plot(x.tail(20), prec['eval_prec1'].tail(20), 'r', label="Top 1")
    for var in (top1):
        p2.annotate('%0.2f, %0.2f' % (max_indx, max_acc), xy=(max_indx, max_acc), xytext=(8, 0), 
            xycoords=('axes fraction', 'data'), textcoords='offset points')
    p2.set_xlabel("Epochs")
    p2.set_ylabel("Validation Accuracy")
    p2.legend(loc='upper right')

    p3.set_title("Top 5: Last 20 epochs")
    p3.plot(x.tail(20), prec['eval_prec5'].tail(20),'b', label="Top 5")
    p3.set_xlabel("Epochs")
    p3.set_ylabel("Validation Accuracy")
    p3.legend(loc='upper right')

    _.subplots_adjust(hspace=1)
    plt.savefig("Validation Accuracy")

def _early_stop(summary):
    loss = summary.tail(20)[['eval_loss']].values

    keep_training = True

    # Counts the number epochs without a new better local minima
    count = 0

    # Max possible error % is 100
    err = 101

    for i in np.nditer(loss):
        # Found new better local minima
        if i < err:
            err = i
            count = 0
        else:
            # Still searching for better local minima
            count = count + 1
        
        # If no better local minima has been found in 20 epochs, stop training
        if count == 20:
            keep_training = False
        
    if keep_training:
        print("No signs of early stopping, keep training!")
    else:
        print("No progress, stop training!")
            

    

# Main
summary = _get_csv()
_plot_error(summary)
_plot_accuracy(summary)
_early_stop(summary)
