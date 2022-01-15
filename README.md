# LiveCELL Cell Line Classifier 

A pipeline for detecting cell type from Microscopy image. Can differentiate between Adenocarcinoma, Breast Cancer, Glioblastoma, Ductal Carcinoma, Neuroblastoma, Microglial and Tumorigenic cells. These cells lines are from the LIVECell dataset, which include A172, BT474, BV2, Huh7, MCF7, SHSY5Y, SkBr3, SKOV3.

This is built with PyTorch. I plan to implement a Vision Tranformer (ViT) architecture as well. 

The code is scaled up to run on multiple GPU's via Kubernetes in the `run.yaml` file. To run on your compute cluster, set up the yaml file appropriately. The credential file contains the AWS key ID, AWS access key, and your CometML API key, respectively. 