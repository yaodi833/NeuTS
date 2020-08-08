import torch
import torch.nn.functional as F
import torch.autograd as autograd
import time

from torch import nn
from tools import config


class Attention(nn.Module):

	def __init__(self, dim):
		super(Attention, self).__init__()
		self.linear_out = nn.Linear(dim * 2, dim)
		self.mask = None
		self.linear_weight = None
		self.linear_bias = None

	def set_mask(self, mask):
		"""
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
		self.mask = mask

	def forward(self, output, context):
		output = autograd.Variable(output, requires_grad=False)
		context = autograd.Variable(context, requires_grad=False)
		batch_size = output.size(0)
		output = output.view(batch_size, 1, -1)
		hidden_size = output.size(2)
		input_size = context.size(1)
		# (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
		# print 'output_size: {}'.format(output.size())
		# print 'context size: {}'.format(context.size())
		# print len(torch.nonzero(context))
		attn = torch.bmm(output, context.transpose(1, 2))
		# print 'attn size: {}'.format(attn.size())
		self.mask = (attn.data == 0).byte()
		if self.mask is not None:
			attn.data.masked_fill_(self.mask, -float('inf'))
		# print attn
		attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
		# (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
		oatt = (attn.data != attn.data).byte()
		attn.data.masked_fill_(oatt, 0.)
		# print len(torch.nonzero(attn != 0))
		mix = torch.bmm(attn, context)

		# print len(torch.nonzero(mix.data != 0))
		# print 'mix size: {}'.format(mix.size())
		# concat -> (batch, out_len, 2*dim)
		combined = torch.cat((mix, output), dim=2)
		# output -> (batch, out_len, dim)

		out = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
		# print output
		out = torch.squeeze(out, 1)
		# print 'out size: {}'.format(out.size())
		# print out
		# print out
		# print out
		return out, attn

	def grid_update_atten(self, output, context):
		output = autograd.Variable(output, requires_grad=False)
		context = autograd.Variable(context, requires_grad=False)
		batch_size = output.size(0)
		output = output.view(batch_size, 1, -1)
		hidden_size = output.size(2)
		input_size = context.size(1)
		# (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
		# print 'output_size: {}'.format(output.size())
		# print 'context size: {}'.format(context.size())
		attn = torch.bmm(output, context.transpose(1, 2))
		# print 'attn size: {}'.format(attn.size())
		self.mask = (attn.data == 0).byte()
		if self.mask is not None:
			attn.data.masked_fill_(self.mask, -float('inf'))
		# print attn
		attn = F.softmax(attn.view(-1, input_size), -1).view(batch_size, -1, input_size)
		# (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
		mix = torch.bmm(attn, context)
		# print mix
		# print 'mix size: {}'.format(mix.size())
		# concat -> (batch, out_len, 2*dim)
		combined = torch.cat((mix, output), dim=2)
		# output -> (batch, out_l
		self.linear_weight = autograd.Variable(self.linear_out.weight.data, requires_grad=False).cuda()
		self.linear_bias = autograd.Variable(self.linear_out.bias.data, requires_grad=False).cuda()

		out = F.tanh(F.linear(combined.view(-1, 2 * hidden_size), self.linear_weight, self.linear_bias)).view(
			batch_size, -1, hidden_size)
		# print output
		out = torch.squeeze(out, 1)
		# print 'out size: {}'.format(out.size())
		# print out
		# print out
		oatt = (out.data != out.data).byte()
		out.data.masked_fill_(oatt, 0.)
		# print out
		return out.cuda(), attn


class SpatialExternalMemory(nn.Module):

	def __init__(self, grid_size, H):
		super(SpatialExternalMemory, self).__init__()
		self.grid_size = grid_size

		# The memory bias allows the heads to learn how to initially address
		# memory locations by content
		# self.memory =  autograd.Variable(torch.Tensor(N, M, H).cuda())
		if config.dimensional == 1:
			self.register_buffer('memory', autograd.Variable(torch.Tensor(grid_size[0], H)))
		elif config.dimensional == 2:
			self.register_buffer('memory', autograd.Variable(torch.Tensor(grid_size[0], grid_size[1], H)))
		elif config.dimensional ==3:
			self.register_buffer('memory', autograd.Variable(torch.Tensor(grid_size[0], grid_size[1], grid_size[2], H)))

		# Initialize memory bias
		nn.init.constant(self.memory, 0.0)

	def reset(self):
		"""Initialize memory from bias, for start-of-sequence."""
		nn.init.constant(self.memory, 0.0)

	def size(self):
		return self.grid_size, self.H

	def find_nearby_grids(self, grid_input, w=config.spatial_width):
		if config.dimensional ==1 :
			grid_x = grid_input[:, 0].data
			grid_x_bd = []
			for i in range(-w, w + 1, 1):
				grid_x_t = F.relu(grid_x + i)
				grid_x_bd.append(grid_x_t)
			grid_x_bd = torch.cat(grid_x_bd, 0)
			t = self.memory[grid_x_bd, :].view(len(grid_x), (2 * w + 1), -1)
			return t
		if config.dimensional ==2 :
			grid_x, grid_y = grid_input[:, 0].data, grid_input[:, 1].data
			tens = []
			# tens = []
			grid_x_bd, grid_y_bd = [], []
			# s = time.time()
			for i in range(-w, w + 1, 1):
				for j in range(-w, w + 1, 1):
					grid_x_t = F.relu(grid_x + i)
					grid_y_t = F.relu(grid_y + j)
					grid_x_bd.append(grid_x_t)
					grid_y_bd.append(grid_y_t)
			grid_x_bd = torch.cat(grid_x_bd, 0)
			grid_y_bd = torch.cat(grid_y_bd, 0)
			# print 'Grid cat time: {}'.format(time.time() -s)
			t = self.memory[grid_x_bd, grid_y_bd, :].view(len(grid_x), (2 * w + 1) * (2 * w + 1), -1)
			return t
		if config.dimensional ==3 :
			grid_x, grid_y, grid_z = grid_input[:, 0].data, grid_input[:, 1].data, grid_input[:, 2].data
			tens = []
			# tens = []
			grid_x_bd, grid_y_bd, grid_z_bd = [], [], []
			# s = time.time()
			for i in range(-w, w + 1, 1):
				for j in range(-w, w + 1, 1):
					for k in range(-w, w + 1, 1):
						grid_x_t = F.relu(grid_x + i)
						grid_y_t = F.relu(grid_y + j)
						grid_z_t = F.relu(grid_y + k)
						grid_x_bd.append(grid_x_t)
						grid_y_bd.append(grid_y_t)
						grid_z_bd.append(grid_z_t)
			grid_x_bd = torch.cat(grid_x_bd, 0)
			grid_y_bd = torch.cat(grid_y_bd, 0)
			grid_z_bd = torch.cat(grid_z_bd, 0)
			# print 'Grid cat time: {}'.format(time.time() -s)
			t = self.memory[grid_x_bd, grid_y_bd,grid_z_bd, :].view(len(grid_x), (2 * w + 1)*(2 * w + 1) * (2 * w +
			                                                                                                1), -1)
			return t

	def update(self, grid_input, updates):
		if config.dimensional ==1 :
			grid_x = grid_input[:, 0].data
			self.memory[grid_x, :]= updates
		elif config.dimensional ==2:
			grid_x, grid_y = grid_input[:, 0].data, grid_input[:, 1].data
			self.memory[grid_x, grid_y, :]= updates
		elif config.dimensional ==3:
			grid_x, grid_y, grid_z = grid_input[:, 0].data, grid_input[:, 1].data, grid_input[:, 1].data
			self.memory[grid_x, grid_y, grid_z, :]= updates

	def read(self, grid_input):
		if config.dimensional ==1 :
			grid_x = grid_input[:, 0].data
			return self.memory[grid_x, :]
		elif config.dimensional ==2:
			grid_x, grid_y = grid_input[:, 0].data, grid_input[:, 1].data
			return self.memory[grid_x, grid_y, :]
		elif config.dimensional ==3:
			grid_x, grid_y, grid_z = grid_input[:, 0].data, grid_input[:, 1].data, grid_input[:, 1].data
			return self.memory[grid_x, grid_y, grid_z, :]
