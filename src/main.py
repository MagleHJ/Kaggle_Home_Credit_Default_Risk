import dataset
from feature_processing import feature_engineering
import models
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = dataset.load_data(debug=True)
df = feature_engineering(df)

feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV']]

models.get_model(df, feats)