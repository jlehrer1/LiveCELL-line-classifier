import pandas as pd 
import numpy as np 
import pandas as pd 
import os 
import pathlib
from sklearn import preprocessing

mapping = {
    "A172": "Glioblastoma",
    "BT474": "Ductal Carcinoma",
    "BV2": "Microglial",
    "Huh7": "Tumorigenic",
    "MCF7": "Breast Cancer",
    "SHSY5Y": "Neuroblastoma",
    "SkBr3": "Adenocarcinoma",
    "SKOV3": "Adenocarcinoma"
}
here = pathlib.Path(__file__).parent.absolute()
le = preprocessing.LabelEncoder()

df = pd.DataFrame(columns=['label'])
df.index.name = 'filename'
for t in mapping:
    for f in os.listdir(os.path.join(here, '..', '..', 'images', t)):
        df.loc[os.path.join(t, f), :] = mapping[t]

df['class'] = le.fit_transform(df['label'])
df.to_csv(os.path.join(here, '..', '..', 'labels.csv'))
