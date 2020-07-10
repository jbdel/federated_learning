Link to the handout : [[link]](https://github.com/jbdel/federated_learning/blob/master/Stanford_federated_handout.pdf) <br/><br/>

## Datasets
**Retina**<br/><br/>
 Link to the Retina dataset: [[link]](https://www.kaggle.com/c/diabetic-retinopathy-detection)<br/>
And preprocess the images according to preprocessImages.py

**ODIR5k**<br/><br/>
ODIR5k train set : [[link]](https://drive.google.com/file/d/1UGrMGfb9zvbBqOvbV62G-XdUlBIAvOad/view) and annotations [[link]](https://drive.google.com/file/d/1jc7Dmp26km0PKRwf9u3Xcyui4SRiojcT/view) <br/><br/>

offsite testing image - [[link]](https://drive.google.com/file/d/19OD9a29nrSbLC2Pch4UZtpp8qtFZLd-y/view

**REFUGE**<br/><br/>	
Images and Glaucoma labels train [[link]](https://www.dropbox.com/s/xd40dewhj0v5gw1/REFUGE-Training400.zip?dl=0) <br/>
validation [[link]](https://www.dropbox.com/s/hhq1srz9ceot8sf/REFUGE-Validation400.zip?dl=0) <br/>
test [[link]](https://www.dropbox.com/s/t1ijw6mdqhd79dm/REFUGE-Test400.zip?dl=0) <br/><br/>

Annotation train : [[link]](https://www.dropbox.com/s/030vecfp36ikiml/Annotation-Training400.zip?dl=0) <br/>
validation : [[link]](https://www.dropbox.com/s/sdgfefzomm5auog/REFUGE-Validation400-GT.zip?dl=0) <br/>
test [[link]](https://www.dropbox.com/s/2w0aof1tqp9gi5a/REFUGE-Test-GT.zip?dl=0)  <br/>


## Training a model
```
- Run `python main.py --central_path ADNI_experiment --data_path Data  --num_inst 2 --inst_id 1 --phase train`
```

```
- Note: data_path 

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

Output:
All the output models/results will be saved on the created sub-folder, sub-folder is same as --central_path paramters.
```
