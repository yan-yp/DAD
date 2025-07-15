## DAD
Related Manuscript: This code is associated with “Progressive Disentanglement for Robust Image Anomaly Detection,” submitted to The Visual Computer (2025). Please cite the paper if you use these resources.

[DOI] 10.5281/zenodo.15901469
## Datasets
To train on the MVTec Anomaly Detection dataset, download the data and extract it. You can run the download_dataset.sh script from the project directory to download the MVTec to the datasets folder.
## Installation
The conda environement used in the project is decsribed in requirements.txt.
## Train the Model
  1. Reference token selection
     ```
     python build_memory.py
     ```
	
 2. Structure prediction
     ```
     python shape_reconstruction.py --gpu_id 0 --obj_id 5 --lr 0.0001 --bs 8 --epochs 700 --data_path ..\datasets\
           mvtec\ --anomaly_source_path ..\datasets\dtd\images\ --checkpoint_path .\shape_repair_checkpoints\ --log_path .\logs\
      ```
     
	
 3. Reconstruction and anomaly detection
     ```
     python train_DAD.py
     ```
## Test the Model
     ```
     python test_DAD.py
     ```
