import numpy as np
import SimpleITK as sitk
import radiomics.featureextractor as featureextractor


def voxelBased(feature_list, images, masks, kernel_size, label, **params):
    """
    Top N개의 feature들에 대해서, RFM을 생성합니다. kernel_size와 label을 정해주세요.
    """
    Res = {}

    params['kernelRadius'] = kernel_size
    params['maskedKernel'] = False
    params['force2D'] = False
    params['label'] = label
    params['initValue'] = np.nan

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    print(f'starting feature extraction with kernel size = {kernel_size}')

    feature_category = {}

    for feature_name in feature_list:
        parts = feature_name.split('_')
        category = parts[1]
        feature = parts[2]
        if category not in feature_category:
            feature_category[category] = []

        feature_category[category].append(feature)


    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(**feature_category)

    if next(iter(images.values())).GetDimension() == 2:
        print('2d Image')
        for file_name in images.keys():
            image = images[file_name]
            mask = masks[file_name]

            image_array = sitk.GetArrayFromImage(image)
            mask_array = sitk.GetArrayFromImage(mask)
        
            image_array_expanded = np.expand_dims(image_array, axis=-1)
            mask_array_expanded = np.expand_dims(mask_array, axis=-1)

            image_expanded = sitk.GetImageFromArray(image_array_expanded)
            mask_expanded = sitk.GetImageFromArray(mask_array_expanded)

            res = extractor.execute(image_expanded, mask_expanded, voxelBased=True)
        
            Res[file_name] = res
            print(f'{file_name} RFM Calculated')
    else:
        print('3d Image')
        for file_name in images.keys():
            image = images[file_name]
            mask = masks[file_name]

            image_array = sitk.GetArrayFromImage(image)
            mask_array = sitk.GetArrayFromImage(mask)

            res = extractor.execute(image, mask, voxelBased=True)

            Res[file_name] = res
            print(f'{file_name} RFM Calculated')
    
    return Res

def makeRFM(Res, feature_list):
    RFM = {}
    for feature_name in feature_list:
        RFM[feature_name] = {}
        for file_name in Res.keys():
            RFM[feature_name][file_name] = sitk.GetArrayFromImage(Res[file_name][feature_name]) # 그 환자의, 특정 feature의 sitk 객체

    return RFM