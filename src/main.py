import dataset
from feature_processing import feature_engineering
import models
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data_list = dataset.load_data()
df = feature_engineering(data_list)

feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV']]

models.get_model(df, feats)