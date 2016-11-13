1. First, run convert_to_gray.py to standardize and convert all images into one giant data frame, 
where each row represents all pixels of each image (64 x 64 gray image after resizing)

2. Run xgb_test.py to train and predict on the dataframe data file generated from step 1, and result will be saved in the same directory.
