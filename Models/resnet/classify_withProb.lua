--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local t = require './datasets/transforms'
local imagenetLabel = {1, 2, 3, 4}

--[[
if #arg < 2 then
   io.stderr:write('Usage: th classify.lua [MODEL] [FILE]...\n')
   os.exit(1)
end
for _, f in ipairs(arg) do
   if not paths.filep(f) then
      io.stderr:write('file not found: ' .. f .. '\n')
      os.exit(1)
   end
end
--]]

-- Load the model
local model = torch.load(arg[1])
local softMaxLayer = cudnn.SoftMax():cuda()

-- add Softmax layer
model:add(softMaxLayer)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

local N = 4

require 'Dataframe'
df = Dataframe('./dsg_data/sample_submission4.csv')
df:add_column('prob', 0)

local image_path = '/../../data/roof_augmented/test/'
for i=1,df:shape()["rows"] do
   local img = image.load(image_path .. df:get_column('Id')[i] .. '.jpg', 3, 'float')
   
   -- Scale, normalize, and crop the image
   img = transform(img)

   -- View as mini-batch of size 1
   local batch = img:view(1, table.unpack(img:size():totable()))
   
   -- Get the output of the softmax
   local output = model:forward(batch:cuda()):squeeze()
   
   -- Get the top 5 class indexes and probabilities
   local probs, indexes = output:topk(N, true, true)
   
   df:update(function(row) return row['Id'] == df:get_column('Id')[i] end,
     function(row) 
	row['label'] = imagenetLabel[indexes[1]] 
	row['prob'] = probs[1]	
	return row end)
   print(i, probs[1], imagenetLabel[indexes[1]])
   print('')
end

df:to_csv('./dsg_output/resnet32_scratch_Epoch32_Val85.03_withProb.csv')
