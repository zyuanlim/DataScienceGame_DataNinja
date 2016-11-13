How to train model:
caffe train -solver ./solver.prototxt -gpu 0 2>&1 | tee train.log

How to get prediction:
python run_caffenet_predict.py
