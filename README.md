[[Link to the handout]](https://github.com/jbdel/federated_learning/blob/master/Stanford_federated_handout.pdf) <br/><br/>
## Datasets
**Retina**<br/><br/>

[[Retina dataset]](https://www.kaggle.com/c/diabetic-retinopathy-detection)<br/>
Preprocess the images according to preprocessImages.py

<b>Preprocess (unint8, 196736 bytes per image)</b><br/>
![preprocess](https://i.imgur.com/2ymMhnA.jpg)

Preprocessed data: [[Dropbox]](https://www.dropbox.com/s/7rraox4puo6vcnx/data.zip?dl=1) [[Sharepoint]](https://alumniumonsac-my.sharepoint.com/:u:/g/personal/532927_umons_ac_be/EZ4cjkHO4pVHq_P3XNbui58BkOiigiNirBDEvYoXQu2Gpg?e=paf2r7) [[Drive]](https://drive.google.com/file/d/1VCJIU3r-qx6etPoHcSJRsJAESyLr3cVE/view?usp=sharing)

```
Label balance for train Counter({0: 25810, 2: 5292, 1: 2443, 3: 873, 4: 708})
Label balance for test Counter({0: 39533, 2: 7861, 1: 3762, 3: 1214, 4: 1206})

11G	data/test
6,6G	data/train
```
[[So far, dataloader loads 100 files]](https://github.com/jbdel/federated_learning/blob/master/retina_dataset.py#L41) <br/>

**ODIR5k**<br/><br/>
ODIR5k train set : [[link]](https://drive.google.com/file/d/1UGrMGfb9zvbBqOvbV62G-XdUlBIAvOad/view) and annotations [[link]](https://drive.google.com/file/d/1jc7Dmp26km0PKRwf9u3Xcyui4SRiojcT/view) <br/><br/>

offsite testing image - [[link]](https://drive.google.com/file/d/19OD9a29nrSbLC2Pch4UZtpp8qtFZLd-y/view)

**REFUGE**<br/><br/>	
Images and Glaucoma labels train [[link]](https://www.dropbox.com/s/xd40dewhj0v5gw1/REFUGE-Training400.zip?dl=0) <br/>
validation [[link]](https://www.dropbox.com/s/hhq1srz9ceot8sf/REFUGE-Validation400.zip?dl=0) <br/>
test [[link]](https://www.dropbox.com/s/t1ijw6mdqhd79dm/REFUGE-Test400.zip?dl=0) <br/><br/>

Annotation train : [[link]](https://www.dropbox.com/s/030vecfp36ikiml/Annotation-Training400.zip?dl=0) <br/>
validation : [[link]](https://www.dropbox.com/s/sdgfefzomm5auog/REFUGE-Validation400-GT.zip?dl=0) <br/>
test [[link]](https://www.dropbox.com/s/2w0aof1tqp9gi5a/REFUGE-Test-GT.zip?dl=0)  <br/>


