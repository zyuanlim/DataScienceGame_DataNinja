th main.lua -retrain pretrained_weights/resnet-34.t7 -data ../../roof_augmented/organised/ -resetClassifier true -nClasses 4 -LR 0.0075 -nEpochs 10 -manualSeed 0 -save './resnet34' -batchSize 10
