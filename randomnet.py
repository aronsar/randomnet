import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


class DataLoader:
  '''
  Objects of this class contain some data of length self.len and can be
  iterated over, returning batch_size length matrices. The format of the
  data is:
  
  data_batch: [batch_size, numerical_data_dim]
  boolean_batch: [batch_size, boolean_data_dim] # includes negations of bools
  label_batch: [batch_size, label_dim] # one hot label vector
  '''
  
  def __init__(self, data_dir, batch_size, is_training):
    self.data = np.load(data_dir + '/data.npz')['data']
    self.boolean = np.load(data_dir + '/bool_mat.npz')['bool_mat']
    self.label = np.load(data_dir + '/label.npz')['label']
    self.batch_size = batch_size
    
    lim = int(0.8 * len(self.label)) # boundary between training and test data
    
    if is_training:
      self.data = self.data[:lim,:]
      self.boolean = self.boolean[:lim,:]
      self.label = self.label[:lim]
      self.len = len(self.label)
      self.max_batch_index = int(self.len / float(self.batch_size))
      
    else: # is test
      self.data = self.data[lim:,:]
      self.boolean = self.boolean[lim:,:]
      self.label = self.label[lim:]
      self.len = len(self.label)
      self.max_batch_index = int(self.len / float(self.batch_size))
    
  def __iter__(self):
    #import pdb; pdb.set_trace()
    self.batch_index = 0
    shuffled_idxs = np.arange(self.len)
    random.shuffle(shuffled_idxs)
    
    self.data = self.data[shuffled_idxs, :]
    self.boolean = self.boolean[shuffled_idxs, :]
    self.label = self.label[shuffled_idxs]
    return self
    
  def __next__(self):
    if self.batch_index < self.max_batch_index:
      # FIXME: what the hell is going on here?
      data_batch    =    torch.from_numpy(self.data[self.batch_index * self.batch_size:(self.batch_index + 1) * self.batch_size, :]).float()
      boolean_batch = torch.from_numpy(self.boolean[self.batch_index * self.batch_size:(self.batch_index + 1) * self.batch_size, :].astype(np.int32)).int()
      label_batch   =   torch.from_numpy(self.label[self.batch_index * self.batch_size:(self.batch_index + 1) * self.batch_size].astype(np.uint8)).long()
      
      self.batch_index += 1
      return (data_batch, boolean_batch, label_batch)
    
    else:
      raise StopIteration
      
  def data_dimensions(self):
    input_dim = self.data.shape[1]
    num_bools = self.boolean.shape[1]
    label_dim = 2 # FIXME: statistically, you could take the max of the label vector
                  # to see what the largest value was, but that wouldn't work for
                  # small batch sizes... ugly situation (but yeah, don't hardcode)
    return (input_dim, num_bools, label_dim)
      
      
class Net(nn.Module):
  '''
  The purpose of this network is to test if randomizing the order of layers in a
  neural network while also assigning each layer to a boolean value will produce
  modular layers that can be arbitrarily combined
  
  Architecture of this network
  
  First layer: maps input to random switching layer size
  Layers in self.layers: randomly shuffled layers, each of which corresponds to a boolean,
    and the value of the boolean determines if "true" layers is used or "false" layer
  Last layer: maps to label size
  '''
  
  def __init__(self, input_dim, random_layer_dim, num_bools, label_dim):
    super(Net, self).__init__()
    
    # defining the layers
    self.first_layer = nn.Linear(input_dim, random_layer_dim)
    
    self.layers = []
    for layer_id in range(num_bools):
      # we append tuples of fc layers, (true layer, false layer, layer id)
      self.layers.append((nn.Linear(random_layer_dim, random_layer_dim), nn.Linear(random_layer_dim, random_layer_dim), layer_id))
      
    self.last_layer = nn.Linear(random_layer_dim, label_dim)

  def forward(self, x, b):
    #import pdb; pdb.set_trace()
    x = F.relu(self.first_layer(x))
    
    # we randomize the layer order
    randomized_layers = random.sample(self.layers, len(self.layers)) 
    for layer in randomized_layers:
      true_layer, false_layer, layer_id = layer # separate layer tuple
      
      # do computation for both values of boolean
      true_x = F.relu(true_layer(x)) 
      false_x = F.relu(false_layer(x))
      #import pdb; pdb.set_trace()
      x = false_x
      
      # we did a clever thing, keeping track of which layer is which using the layer id,
      # and using the layer id to index the correct column of the boolean matrix
      bool_col = layer_id 
      bool_vec = b[:,bool_col].byte() # PICK ME UP: convert bool_col to long
      x[bool_vec] = true_x[bool_vec]
      
    logits = self.last_layer(x)
    
    return logits

    
def train(args, model, device, train_loader, optimizer, epoch):
  model.train() # tells model we are about to train it
  for batch_idx, (data, boolean, label) in enumerate(train_loader):
    data, boolean, label = data.to(device), boolean.to(device), label.to(device)
    optimizer.zero_grad()
    output = model(data, boolean)
    #import pdb; pdb.set_trace()
    loss = F.cross_entropy(output, label)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), train_loader.len, loss.item()))


def test(args, model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, boolean, label in test_loader:
      data, boolean, label = data.to(device), boolean.to(device), label.to(device)
      output = model(data, boolean)
      test_loss += F.cross_entropy(output, label, reduction='sum').item() # sum up batch loss
      pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
      correct += pred.eq(label.view_as(pred)).sum().item()

  test_loss /= test_loader.len
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, test_loader.len,
    100. * correct / test_loader.len))

    
def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
            help='input batch size for training (default: )')
            
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
            
  parser.add_argument('--epochs', type=int, default=40, metavar='N',
            help='number of training epochs (default: 10)')
            
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
            
  parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
            
  parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
            
  parser.add_argument('--log-interval', type=int, default=2, metavar='N',
            help='how many batches to wait before logging training status')
            
  parser.add_argument('--data-dir', type=str, default='./data_simple', metavar='ddir',
            help='the location of the data')
            
  args = parser.parse_args()
  # FIXME: use cuda please
  #use_cuda = not args.no_cuda and torch.cuda.is_available()
  use_cuda = False
  torch.manual_seed(args.seed)

  device = torch.device("cuda" if use_cuda else "cpu")

  train_data = DataLoader(args.data_dir, args.batch_size, is_training=True)
  test_data = DataLoader(args.data_dir, args.batch_size, is_training=False)
  
  input_dim, num_bools, label_dim = train_data.data_dimensions()
  random_layer_dim = 10
  model = Net(input_dim, random_layer_dim, num_bools, label_dim).to(device)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_data, optimizer, epoch)
    test(args, model, device, test_data)


if __name__ == '__main__':
  main()