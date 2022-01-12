import pandas as pd 
import numpy as np 
import pandas as pd 
import os 
import pathlib
from sklearn import preprocessing
import boto3

def download_images(here, labels):
    with open(os.path.join(here, '..', '..', 'credentials')) as f:
        key, secret = [line.strip() for line in f.readlines()]

    # create all the folders we need to for the data download
    pathlib.Path(os.path.join(here, '..', '..', 'images')).mkdir(exist_ok=True)
    for label in labels:
        pathlib.Path(os.path.join(here, '..', '..', 'images', label)).mkdir(exist_ok=True)

    # Now we download the files from S3 recursively into the folder 
    s3 = boto3.resource(
        's3',
        endpoint_url="https://s3.nautilus.optiputer.net",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )

    # Get list of objects in the bucket, taking the string after the 8th index removes the 'jlehrer/livecell' at the beginning
    objs = [f.key[16 + 1:] for f in s3.Bucket('braingeneersdev').objects.filter(Prefix="jlehrer/livecell")]
    # Now we download the objects into the correct folder 
    print('Downloading files from S3')
    for file in objs:
        print(f'Downloading {file}')
        s3.Bucket('braingeneersdev').download_file(
            Key=os.path.join('jlehrer', 'livecell', file),
            Filename=os.path.join(here, '..', '..', 'images', file),
        )
    print('Finished downloading files from S3')

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

# First, we download the data with the function defined above
download_images(here, mapping.keys())

le = preprocessing.LabelEncoder()

print('Generating label file for cell lineage classification')
# Generate the labels dataframe
df = pd.DataFrame(columns=['label'])
df.index.name = 'filename'
for t in mapping:
    for f in os.listdir(os.path.join(here, '..', '..', 'images', t)):
        df.loc[os.path.join(t, f), :] = mapping[t]

df['class'] = le.fit_transform(df['label'])
df.to_csv(os.path.join(here, '..', '..', 'labels.csv'))
