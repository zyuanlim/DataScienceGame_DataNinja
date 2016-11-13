```
This script will write all image file names and labels in validation folder (under organised) into a dataframe called validation,
then save it out as validation.csv.
```

import os
import pandas as pd


val_images = {}
path_to_validation = "./roof_augmented/organised/val"
for i in range(1,5):
    for img in os.listdir(os.path.join(path_to_organised, "label%s" %i)):
        val_images[img] = i

validation = pd.DataFrame.from_dict(val_images, orient='index').reset_index()
validation.columns = ['Id', 'label']

validation.to_csv('./validation.csv')
