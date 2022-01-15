import pathlib, os 
import comet_ml
import torch 
import pytorch_lightning as pl 
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader

from data.dataset import CellDataset
from models.lineage_classification import Net

if __name__ == "__main__":
    here = pathlib.Path(__file__).parent.absolute()
    key = [f.rstrip() for f in open(os.path.join(here, '..', 'credentials'))][2]

    dataset = CellDataset(
        images_path=os.path.join(here, '..', 'images'),
        label_path=os.path.join(here, '..', 'labels', 'labels.csv'),
    )
    
    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    traindata = DataLoader(train, batch_size=8, num_workers=32)
    valdata = DataLoader(test, batch_size=8, num_workers=32)

    comet_logger = CometLogger(
        api_key=key,
        project_name="lineage-classifier",  # Optional
        workspace="jlehrer1",
    )

    trainer = pl.Trainer(
        gpus=2, 
        strategy="ddp",
        auto_lr_find=True,
        max_epochs=100000, 
        logger=comet_logger,
    )

    model = Net()
    trainer.fit(model, traindata, valdata)