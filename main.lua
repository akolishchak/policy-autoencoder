--
--  main.lua
--  policy-autoencoder
--
--  Learning transition model and policy in a grid world.
--
--  Created by Andrey Kolishchak on 08/27/16.
--
require 'nn'
require 'optim'
require 'dataset'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Policy Autoencoder')
cmd:text()
cmd:text('Options')
cmd:option('-train_size', 100000, 'size of train set')
cmd:option('-test_size', 10, 'size of test set')
cmd:option('-grid_width', 10, 'grid size')
cmd:option('-max_epoch', 100, 'number of full passes through the training data')
cmd:option('-dropout', 0.5, 'dropout')
cmd:option('-learning_rate', 1e-4, 'learning rate')
cmd:option('-batch_size', 100, 'number of sequences to train on in parallel')
cmd:option('-gpu',1,'0 - cpu, 1 - cunn, 2 - cudnn')
cmd:option('-output_path', 'images', 'path for output images')

local opt = cmd:parse(arg)
opt.grid_size = opt.grid_width*opt.grid_width
opt.action_size = 9

if opt.gpu > 0 then
  require 'cunn'
  if opt.gpu == 2 then
    require 'cudnn'
  end
end

--
-- load data
--
print("loading data...")
local dataset = load_data(opt)
--
-- build model
--
print("building model...")

-- 1:up 2:up-right 3:right 4:down-right 5:down 6:down-left 7:left 8:up-left 9:stop
local action_offset = torch.LongTensor({
        {-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}, {0, 0}
      }):repeatTensor(opt.batch_size, 1, 1)


local encoder = nn.Sequential()
encoder:add(nn.JoinTable(2))
encoder:add(nn.Linear(opt.action_size + opt.grid_size, opt.grid_size))
encoder:add(nn.ReLU())
encoder:add(nn.BatchNormalization(opt.grid_size))
if opt.dropout > 0 then encoder:add(nn.Dropout(opt.dropout)) end
encoder:add(nn.Linear(opt.grid_size, opt.grid_size))
encoder:add(nn.ReLU())
encoder:add(nn.SoftMax())

local decoder = nn.Sequential()
decoder:add(nn.JoinTable(2))
decoder:add(nn.Linear(opt.grid_size + opt.grid_size, opt.action_size))
decoder:add(nn.ReLU())
decoder:add(nn.BatchNormalization(opt.action_size))
if opt.dropout > 0 then decoder:add(nn.Dropout(opt.dropout)) end
decoder:add(nn.Linear(opt.action_size, opt.action_size))
decoder:add(nn.ReLU())
decoder:add(nn.SoftMax())



local model = nn.Sequential()
model:add(nn.ConcatTable()
            :add(encoder)
            :add(nn.SelectTable(-1))
         )
model:add(nn.ConcatTable()
            :add(nn.SelectTable(1))
            :add(decoder)
         )

local criterion = nn.ParallelCriterion()
                    :add(nn.BCECriterion(), 1)
                    :add(nn.BCECriterion(), 1)

print(model)

if opt.gpu > 0 then
  model:cuda()
  criterion:cuda()
  if opt.gpu == 2 then
    cudnn.convert(model, cudnn)
    cudnn.benchmark = true
  end
end

local params, grad_params = model:getParameters()


--
-- optimize
--
local iterations = opt.max_epoch*opt.train_size/opt.batch_size
local batch_start = 1

function feval(x)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()
  
    -- load batch
  local state1 = dataset.train_x.data1[{{batch_start, batch_start+opt.batch_size-1},{}}]
  local state2 = dataset.train_x.data2[{{batch_start, batch_start+opt.batch_size-1},{}}]
  local action = dataset.train_x.action[{{batch_start, batch_start+opt.batch_size-1},{}}]
  
  local input = { action, state1 }
  local target = { state2, action }
  
  -- forward pass
  local output = model:forward(input)
  local loss = criterion:forward(output, target)
  
  -- backward pass
  local dloss_doutput = criterion:backward(output, target)
  model:backward(input, dloss_doutput)
    
  return loss, grad_params
end


--
-- training
--
model:training()
local optim_state = {learningRate = opt.learning_rate}
print("trainig...")

for it = 1,iterations do
  
    local _, loss = optim.adam(feval, params, optim_state)

    if it % 100 == 0 then
      print(string.format("batch = %d, loss = %.12f", it, loss[1]))
    end
  
    batch_start = batch_start + opt.batch_size
    if batch_start > opt.train_size then
      batch_start = 1
    end 
    
end

print("evaluating...")
model:evaluate()

local state1 = dataset.test_x.data1
local state2 = dataset.test_x.data2
local action = dataset.test_x.action
  
local input = { action, state1 }
local target = { state2, action }
  
local output = model:forward(input)
local loss = criterion:forward(output, target)

print(string.format("testing loss = %.12f", loss))

--[[
local pred_action = decoder:forward{state2, state1}
_,action_i = torch.max(action, 2)
_,pred_action_i = torch.max(pred_action, 2)
print(action_i)
print(pred_action_i)
print(pred_action)
local pred_state2 = output[1]
_,state2_i = torch.max(state2, 2)
_,pred_state2_i = torch.max(pred_state2, 2)
print(state2_i)
print(pred_state2_i)
print(pred_state2)
print(state2)
]]--



