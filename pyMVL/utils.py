import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

def topFeatures(radiomics_csv_path, shap_csv_path, top_no):
    """Global radiomics Value and shap value csv to get top n mean shap value
     -> output : list which contains top n features """
    radiomics_df = pd.read_csv(radiomics_csv_path)
    shap_df = pd.read_csv(shap_csv_path)
    cleaned_shap_df = shap_df.loc[:, shap_df.columns.str.startswith('original')]

    mean_abs_shap_values = cleaned_shap_df.abs().mean()
    sorted_mean_abs_values = mean_abs_shap_values.sort_values(ascending=False)
    index_list = sorted_mean_abs_values.index.tolist()

    return index_list[:top_no]

def get_heatmap(extractor, file_name):

    top_no = len(extractor.feature_list)
    Heatmap = np.zeros_like(extractor.RCM[extractor.feature_list[0]][file_name])

    for feature_name in extractor.feature_list:
        Heatmap += extractor.RCM[feature_name][file_name] 
    return Heatmap

def visualize_heatmap(extractor, file_name, heatmap_scale=(-3,3)):

    top_no = len(extractor.feature_list)

    cmap = plt.get_cmap('inferno')
    cmap.set_bad(color='black')
    Heatmap = np.zeros_like(extractor.RCM[extractor.feature_list[0]][file_name])

    for feature_name in extractor.feature_list:
        Heatmap += extractor.RCM[feature_name][file_name] 

    plt.imshow(Heatmap, cmap=cmap, vmin=heatmap_scale[0], vmax=heatmap_scale[1])
    plt.title(f'RCM of {file_name}', fontsize=14)
    plt.colorbar(orientation='vertical', shrink=0.5)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_heatmap_Binary(extractor, file_name, threshold=0, heatmap_scale=(-3,3)):

    top_no = len(extractor.feature_list)

    cmap = plt.get_cmap('inferno')
    cmap.set_bad(color='black')
    Heatmap = np.zeros_like(extractor.RCM[extractor.feature_list[0]][file_name])

    for feature_name in extractor.feature_list:
        Heatmap += extractor.RCM[feature_name][file_name] 
    Heatmap = cv2.GaussianBlur(Heatmap, (7, 7), 0)
    Heatmap = np.where(Heatmap > threshold, Heatmap, np.nan)
    plt.imshow(Heatmap, cmap=cmap, vmin=heatmap_scale[0], vmax=heatmap_scale[1])
    plt.title(f'{file_name} Binarized Heatmap / Threshold = {threshold}', fontsize=14)
    plt.colorbar(orientation='vertical', shrink=0.5)
    plt.axis('off')

    plt.tight_layout()
    plt.show()