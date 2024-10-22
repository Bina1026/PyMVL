from .RCM import *
from .RFM import *
from .utils import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Extractor:
    def __init__(self, radiomics_csv_path, shap_csv_path, top_no:int):
        feature_list = topFeatures(radiomics_csv_path, shap_csv_path, top_no)
        print(feature_list)

        self.feature_list = feature_list
        self.radiomics_df = pd.read_csv(radiomics_csv_path)
        self.shap_df = pd.read_csv(shap_csv_path)

        self.Res = None
        self.RFM = None
        self.RCM = None

    def execute(self, images, masks, kernel_size, label, params, Fast=False):

        print(f'Image Count = {len(images)}, Calculating Radiomics Feature Map..')
        Res = voxelBased(self.feature_list, images, masks, kernel_size, label, **params) # Res[file_name] = ordered dictionary which contains each RFM of images
        self.Res = Res
        self.make_RFM()
        self.make_RCM(Fast)
    
    def make_RFM(self):

        RFM = makeRFM(self.Res, self.feature_list)
        print(f'Radiomics Feature Map Calculated, Mapping shap values to acquire Radiomics Contributional Map..')
        self.RFM = RFM

    def make_RCM(self, Fast):
        
        RCM = makeRCM(self.RFM, self.radiomics_df, self.shap_df, Fast)
        print(f'Radiomics Contributional Map Calculated.')
        self.RCM = RCM

    def visualize_interpolation(self):
        top_no = len(self.feature_list)
        xs = np.arange(0, 1, 0.01)

        fig, axes = plt.subplots(1, top_no, figsize=(25,5))

        axes = axes.flatten()

        scaler = MinMaxScaler()
        
        for ax, feature_name in zip(axes, self.feature_list):
            radiomics_value = self.radiomics_df[feature_name].to_numpy()
            radiomics_value = scaler.fit_transform(radiomics_value.reshape(-1, 1)).reshape(radiomics_value.shape)
            shap_value = self.shap_df[feature_name].to_numpy()
            ax.scatter(radiomics_value, shap_value, color='red')
            lin = interpolate_shap(feature_name, self.radiomics_df, self.shap_df)
            ax.plot(xs, lin(xs))
            ax.set_title(feature_name, fontsize=12)

            
    def visualize_heatmap(self, file_name, feature_scale=(-1,1), heatmap_scale=(-3,3)):

        top_no = len(self.feature_list)

        cmap = plt.get_cmap('jet')
        cmap.set_bad(color='black')
        Heatmap = np.zeros_like(self.RCM[self.feature_list[0]][file_name])

        for feature_name in self.feature_list:
            Heatmap += self.RCM[feature_name][file_name] 

        fig, axes = plt.subplots(1, top_no+1, figsize=(25, 5))

        axes = axes.flatten()
    
        for ax, (key, value) in zip(axes, self.RCM.items()):
            cax = ax.imshow(value[file_name], cmap=cmap, vmin=feature_scale[0], vmax=feature_scale[1])
            ax.set_title(key, fontsize=8)
            fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.5)
            ax.axis('off')
 
        cax = axes[-1].imshow(Heatmap, cmap=cmap, vmin=heatmap_scale[0], vmax=heatmap_scale[1])
        axes[-1].set_title('Heatmap', fontsize=8)
        fig.colorbar(cax, ax=axes[-1], orientation='vertical', shrink=0.5)
        axes[-1].axis('off')

        plt.tight_layout()
        plt.show()

 

