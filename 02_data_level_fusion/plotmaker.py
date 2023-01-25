# -*- coding: utf-8 -*-
"""
This is the file used to create plots from the running logs of the filtering/upsampling task

@author: daniel
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#%% A function to save plots

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "plots")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=resolution)
    
    
#%% Plotting 

def get_ssd_type_plot(df, name, metric, group1, group2):
    # Group by image name and average out the columns
    df_name = df[df['name']==name].groupby([group1, group2]).mean().reset_index()
    
    # Create and save plot
    plt.figure(figsize=(14,8))
    
    # Title and axes labels
    plt.title('Average ' + metric + ' in case of ' + name + ' estimation', size=16)
    plt.xlabel('Image name', size=14)
    plt.xticks(rotation=90)
    plt.ylabel(metric, size=14)
    
    if(metric == 'ssim'):
        plt.ylim([0.8, 1.0])
    
    sns.barplot(data=df_name, x=group1, y=metric, hue=group2)
    save_fig(name + '_' + metric + '_' + group2 + '_figure')
    plt.show()


#%%

if(__name__ == '__main__'):
    # Reading the logs into dataframes
    df_log = pd.read_csv('runlogs.csv', sep=';', header=0)
    df_sig = pd.read_csv('bestparams_sig.csv', sep=';', header=0)
    df_win = pd.read_csv('bestparams_win.csv', sep=';', header=0)
    
    # Some preprocessing for the useless logs
    df_log = df_log[df_log["window_size"]!=13]
    df_log = df_log[df_log["image_name"].str.contains("aloe2")==False]
    #df_log = df_log[df_log["image_name"].str.contains("backpack")==False]
    #df_log = df_log[df_log["image_name"].str.contains("bicycle")==False]

    print('Creating plots for image similarity metrics...')
    for name in ['iterative','naive']:
        for metric in ['ssd','ssim','ncc','time']:
            get_ssd_type_plot(df_log, name, metric, 'image_name', 'window_size')
            get_ssd_type_plot(df_log, name, metric, 'image_name', 'sigma')
