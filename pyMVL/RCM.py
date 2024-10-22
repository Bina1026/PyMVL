import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

def interpolate_shap(feature_name, radiomics_df, shap_df):
    scaler = MinMaxScaler()
    radiomics_value = radiomics_df[feature_name].to_numpy()
    radiomics_value = scaler.fit_transform(radiomics_value.reshape(-1, 1)).reshape(radiomics_value.shape)
    shap_value = shap_df[feature_name].to_numpy()

    lin = interp1d(radiomics_value, shap_value, fill_value='extrapolate')
    return lin

def replace_threshold(arr, upper_bound, lower_bound):
    arr[arr > upper_bound] = upper_bound
    arr[arr < lower_bound] = lower_bound
    return arr

def replaceOutlierandNormalize(dict_image: dict, Fast = False) -> dict:
    """
    Function for outlier imputation of local radiomics values for a specific radiomics feature's RFM.
    
    Args:
        radiomics_data (Dict[str, float]): A dictionary containing radiomics values.
    
    Returns:
        Dict[str, float]: A dictionary with outliers imputed.
    """

    if Fast == True:
        print('Fast mode activated')
        kys = list(dict_image.keys())
        selected_kys = np.random.choice(kys, int(0.1 * len(kys)), replace=False)
        all_values = np.concatenate([dict_image[file_name][~np.isnan(dict_image[file_name])].flatten() for file_name in selected_kys])
        Q0 = np.percentile(all_values, 0)
        Q1 = np.percentile(all_values, 25)
        Q3 = np.percentile(all_values, 75)
        Q4 = np.percentile(all_values, 100)

        IQR = Q3 - Q1
        upper_outlier = Q3 + 1.5 * IQR
        lower_outlier = Q1 - 1.5 * IQR
        

        replaced_dict = {file_name: replace_threshold(dict_image[file_name], upper_outlier, lower_outlier) for file_name in dict_image.keys()}

        Min = 0
        Max = 0

        if upper_outlier > Q4:
            Max = Q4
        else:
            Max = upper_outlier

        if lower_outlier < Q0:
            Min = Q0
        else:
            Min = lower_outlier

        replaced_dict2 = {file_name: (replaced_dict[file_name] - Min) / (Max - Min) for file_name in replaced_dict.keys()}

    elif Fast == False:
        print('fast mode deactivated')
        all_values = np.concatenate([dict_image[file_name][~np.isnan(dict_image[file_name])].flatten() for file_name in dict_image.keys()])
        Q0 = np.percentile(all_values, 0)
        Q1 = np.percentile(all_values, 25)
        Q3 = np.percentile(all_values, 75)
        Q4 = np.percentile(all_values, 100)

        IQR = Q3 - Q1
        upper_outlier = Q3 + 1.5 * IQR
        lower_outlier = Q1 - 1.5 * IQR
        

        replaced_dict = {file_name: replace_threshold(dict_image[file_name], upper_outlier, lower_outlier) for file_name in dict_image.keys()}

        Min = 0
        Max = 0

        if upper_outlier > Q4:
            Max = Q4
        else:
            Max = upper_outlier

        if lower_outlier < Q0:
            Min = Q0
        else:
            Min = lower_outlier

        replaced_dict2 = {file_name: (replaced_dict[file_name] - Min) / (Max - Min) for file_name in replaced_dict.keys()}
    return replaced_dict2
       
def shapMapping(dict_image: dict, feature_name, radiomics_df, shap_df, Fast=False): 
    """
    특정 radiomics feature에 대한, RFM과, feature name을 받아서, shap mapping된 RCM 반환
    """

    replaced_dict = replaceOutlierandNormalize(dict_image, Fast)
    lin = interpolate_shap(feature_name, radiomics_df, shap_df)

    Mapping = {file_name: lin(replaced_dict[file_name]) for file_name in replaced_dict.keys()}

    return Mapping

def makeRCM(RFM, radiomics_df, shap_df, Fast=False):
    RCM = {}
    for feature_name in RFM.keys():
        Mapping = shapMapping(RFM[feature_name], feature_name, radiomics_df, shap_df, Fast)
        RCM[feature_name] = Mapping

    return RCM