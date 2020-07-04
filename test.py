import os
import pydicom
import pandas as pd

dir = '/home/vicky/Research/Deep_Learning/Pytorch/distrubuted/Code_Deploy_CWT/Data/'
df = pd.read_csv('/home/vicky/Research/Deep_Learning/Pytorch/distrubuted/Code_Deploy_CWT/Data/labels_v.csv')
df = df.drop_duplicates()
patient_IDs = []
images = []
labels = []
for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
        if fname.endswith('.dcm'):
            path = os.path.join(root, fname)
            dicoms = pydicom.read_file(path)
            images.append(path)

            patient_IDs.append(dicoms.SOPInstanceUID)
            labels.append(df[df['ID'] == dicoms.SOPInstanceUID]['Label'].values[0])
net_names =[os.path.basename(name) for name in images]
df1 = pd.DataFrame(columns=['ID', 'Label'])

df1['ID'] = net_names
df1['Label'] = labels
df1.to_csv('/home/vicky/Research/Deep_Learning/Pytorch/distrubuted/Code_Deploy_CWT/Data/labels.csv', index = False)