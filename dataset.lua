--
--  dataset.lua
--  policy-autoencoder
--
--  Learning transition model and policy in a grid world.
--
--  Created by Andrey Kolishchak on 08/27/16.
--
function get_dot_grid(set_size, grid_size, grid_width, gpu)
  local set = {}
  
  -- initial state
  set.data1 = torch.zeros(set_size, grid_size)
  local rand_idx = (torch.rand(set_size, 1)*grid_size+1):long()
  set.data1:scatter(2, rand_idx, 1)
  
  set.loc1 = torch.Tensor(set_size ,2)
  set.loc1[{{}, 1}] = torch.add(rand_idx, -1):div(grid_width):add(1)
  set.loc1[{{}, 2}] = torch.add(rand_idx, -1):remainder(grid_width):add(1)
  
  -- action
  -- 1:up 2:up-right 3:right 4:down-right 5:down 6:down-left 7:left 8:up-left 9:stop
  local action_offset = torch.LongTensor({
        {-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}, {0, 0}
      }):repeatTensor(set_size, 1, 1)
  
  --set.action = 
  -- take action
  local action = (torch.rand(set_size, 1)*9+1):long()
  local index = torch.LongTensor(set_size,1,2)
  index[{{},{},{1}}] = action
  index[{{},{},{2}}] = action
  set.loc2 = torch.add(set.loc1, action_offset:gather(2, index):double()):clamp(1, grid_width)
  
  set.data2 = torch.zeros(set_size, grid_size)
  set.data2:scatter(2, torch.add(set.loc2[{{}, 1}], -1):mul(grid_width):add(set.loc2[{{}, 2}]):view(-1,1):long(), 1)
  set.action = torch.zeros(set_size, 9):long()
  set.action:scatter(2, action, 1)
  
  if gpu > 0 then
    set.data1 = set.data1:cuda()
    set.data2 = set.data2:cuda()
    set.loc1 = set.loc1:cuda()
    set.loc2 = set.loc2:cuda()
    set.action = set.action:cuda()
  end
 
  return set
end

function load_data(opt)
  local dataset = {}
  
  torch.manualSeed(1)
  if opt.gpu > 0 then
    cutorch.manualSeedAll(1)
  end
  
  dataset.train_x = get_dot_grid(opt.train_size, opt.grid_size, opt.grid_width, opt.gpu)
  dataset.train_y = get_dot_grid(opt.train_size, opt.grid_size, opt.grid_width, opt.gpu)
  
  dataset.test_x = get_dot_grid(opt.test_size, opt.grid_size, opt.grid_width, opt.gpu)
  dataset.test_y = get_dot_grid(opt.test_size, opt.grid_size, opt.grid_width, opt.gpu)
  
  return dataset
end
