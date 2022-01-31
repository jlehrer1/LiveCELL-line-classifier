import pathlib 
import urllib.request
import os 

here = pathlib.Path(__file__).parent.absolute()

def download_labels():
    urls = {
        "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json" :  "train.json",
        "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json" : "val.json",
        "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json" : "test.json",
    }

    if all([os.path.isfile(f) for f in urls.values()]):
        print('Image masks already downloaded, continuing ...')
        return

    for url, fname in urls.items():
        urllib.request.urlretrieve(
            url,
            os.path.join(here, '..', '..', 'images', fname)
        )  

def download_data():
    urls = {
        
    }

if __name__ == "__main__":
    download_data()
    download_labels()