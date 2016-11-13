python make_list.py --recursive 1 ../../roof_augmented/organised/train train # --train_ratio 0.75
python make_list.py --recursive 1 ../../roof_augmented/organised/val val

im2rec_path=~/mxnet/bin/im2rec

python ~/mxnet/tools/im2rec.py train.lst ../../roof_augmented/organised/train --resize=224 --color=1 --encoding='.jpg'
python ~/mxnet/tools/im2rec.py val.lst ../../roof_augmented/organised/val --resize=224 --color=1 --encoding='.jpg'

