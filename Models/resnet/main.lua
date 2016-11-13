--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'optim'
require 'nn'
cutorch.setDevice(1)
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

-- Create loggers
if not paths.dirp(opt.save) then paths.mkdir(opt.save) end
   trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
   testLogger  = optim.Logger(paths.concat(opt.save, 'test.log'))


if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)
   trainLogger:add{
         ['% top1 accuracy (train set)'] = 100 - trainTop1,
         --['% top5 accuracy (train set)'] = 100 - trainTop5,
            ['avg loss (train set)'] = trainLoss
         }
   -- Run model on validation set
   local testTop1, testTop5, testLoss = trainer:test(epoch, valLoader)
   testLogger:add{
         ['% top1 accuracy (test set)'] = 100 - testTop1,
         --['% top5 accuracy (test set)'] = 100 - testTop5,
            ['avg loss (test set)'] = testLoss
         }

   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      print(' * Best model ', testTop1, testTop5)
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt.save)
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
