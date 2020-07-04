1)
Link to the Retina dataset: https://www.kaggle.com/c/diabetic-retinopathy-detection
And preprocess the images according to preprocessImages.py




%.................................................................................................................
Docker usage:
- sudo docker pull vickyqu0/cwt:latest
- docker run cwt --num_inst 2 --inst_id 1 --central_path ADNI_experiment



%.................................................................................................................
Usage:

Training a model
- Run `python main.py --central_path ADNI_experiment --data_path Data  --num_inst 2 --inst_id 1 --phase train`

- Note: data_path 

%.................................................................................................................
Dataset:
- Required data organization (no other folders/files than the ones listed below):
	data_dir/
		labels.csv with following format in each line:
			filename.dcm,label(int from 0 to num_classes-1 or float for regression)
		train/
			filename.dcm files for train
		val/
			filename.dcm files for val
		test/
			filename.dcm files for test
- I put sample dataset in the sub-folder ./Data/.

%.................................................................................................................
Output:
All the output models/results will be saved on the created sub-folder, sub-folder is same as --central_path paramters.
