```
.
├── data
│   ├── images
│   ├── testLabels.csv
│   ├── trainLabels.csv
│   ├── train_liang.csv
│   └── val_liang.csv
├── main.py
├── preprocess.py
├── README.md
├── retina_dataset.py
└── Stanford_federated_handout.pdf
```

Place all .npy files from retina dataset (links below) in the data/images folder.
```
python main.py --batch_size 16 --rounds 5 --sites 4 --samples_site 1000 --epoch_per 1 --distribution 0.5 0.5
```

<b>Results of the script: (homogeneous)</b><br/>
  
3 rounds, 1000 samples/site, 1 epoch per site

```
[Round  1][Site  4][Epoch  1][Step   93/  93] Loss: 0.4973, Lr: 1.00e-04          Evaluation accuracy 68.46666666666667
[Round  2][Site  4][Epoch  1][Step   93/  93] Loss: 0.3272, Lr: 1.00e-04          Evaluation accuracy 76.43333333333334
[Round  3][Site  4][Epoch  1][Step   93/  93] Loss: 0.1500, Lr: 1.00e-04          Evaluation accuracy 72.76666666666667
```

5 rounds, 1000 samples/site, 1 epoch per site
```
[Round  1][Site  4][Epoch  1][Step   62/  62] Loss: 0.2833, Lr: 1.00e-04          Evaluation accuracy 64.26666666666667
[Round  2][Site  4][Epoch  1][Step   62/  62] Loss: 0.2774, Lr: 1.00e-04          Evaluation accuracy 75.0
[Round  3][Site  4][Epoch  1][Step   62/  62] Loss: 0.2409, Lr: 1.00e-04          Evaluation accuracy 75.73333333333333
[Round  4][Site  4][Epoch  1][Step   62/  62] Loss: 0.1499, Lr: 1.00e-04          Evaluation accuracy 72.46666666666667
[Round  5][Site  4][Epoch  1][Step   62/  62] Loss: 0.1425, Lr: 1.00e-04          Evaluation accuracy 77.03333333333333
```

5 rounds, 1500 samples/site, 1 epoch per site
```
[Round  1][Site  4][Epoch  1][Step   93/  93] Loss: 0.3132, Lr: 1.00e-04          Evaluation accuracy 69.63333333333334
[Round  2][Site  4][Epoch  1][Step   93/  93] Loss: 0.2224, Lr: 1.00e-04          Evaluation accuracy 73.83333333333333
[Round  3][Site  4][Epoch  1][Step   93/  93] Loss: 0.2381, Lr: 1.00e-04          Evaluation accuracy 73.7
[Round  4][Site  4][Epoch  1][Step   93/  93] Loss: 0.2268, Lr: 1.00e-04          Evaluation accuracy 78.16666666666666
[Round  5][Site  4][Epoch  1][Step   93/  93] Loss: 0.1443, Lr: 1.00e-04          Evaluation accuracy 71.93333333333334
```

5 rounds, 1500 samples/site, 5 epoch per site
```
[Round  1][Site  4][Epoch  5][Step   93/  93] Loss: 0.1605, Lr: 1.00e-04          Evaluation accuracy 68.83333333333333
[Round  2][Site  4][Epoch  5][Step   93/  93] Loss: 0.0349, Lr: 1.00e-04          Evaluation accuracy 77.26666666666667
[Round  3][Site  4][Epoch  5][Step   93/  93] Loss: 0.1368, Lr: 1.00e-04          Evaluation accuracy 77.13333333333333
[Round  4][Site  4][Epoch  5][Step   93/  93] Loss: 0.0200, Lr: 1.00e-04          Evaluation accuracy 75.2
[Round  5][Site  4][Epoch  5][Step   93/  93] Loss: 0.0934, Lr: 1.00e-04          Evaluation accuracy 76.6
```

[[Link to the handout]](https://github.com/jbdel/federated_learning/blob/master/Stanford_federated_handout.pdf) <br/><br/>
## Datasets
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


