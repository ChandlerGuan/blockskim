import torch
import torch.nn as nn
from typing import List

class BlockSkim(nn.Module):
    def __init__(self, config):
        super(BlockSkim, self).__init__()

        self.kernel_size = 3
        self.stride = 1
        self.pool_kernel_size = 2

        self.block_size = config.block_size
        self.num_attention_heads = config.num_attention_heads

        self.pruned_heads = set()

        self.conv1 = torch.nn.Conv2d(config.num_attention_heads,config.num_attention_heads,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.kernel_size//2
                                        )
        self.bn1 = torch.nn.BatchNorm2d(config.num_attention_heads)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = torch.nn.AvgPool2d(self.pool_kernel_size,self.pool_kernel_size)
        self.conv2 = torch.nn.Conv2d(config.num_attention_heads,config.num_attention_heads,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.kernel_size//2
                                        )
        self.bn2 =  torch.nn.BatchNorm2d(config.num_attention_heads)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = torch.nn.AvgPool2d(self.pool_kernel_size,self.pool_kernel_size)
        self.conv3 = torch.nn.Conv2d(config.num_attention_heads,1,
                                        kernel_size=1,
                                        stride=1
                                        )
        self.relu3 = nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(self.num_attention_heads*self.block_size*self.block_size//16,2)

    def prune_heads(self, heads: List):
        if len(heads) == 0:
            return
        if self.pruned_heads:
            return

        mask = torch.ones(self.num_attention_heads)
        for head in heads:
            mask[head] = 0
        mask = torch.arange(self.num_attention_heads,device=self.conv1.weight.device,)[mask==1]

        self.prev_num_attention_heads = self.num_attention_heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)
        
        new_conv1 = torch.nn.Conv2d(self.num_attention_heads,self.prev_num_attention_heads,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=self.kernel_size//2,
                                device=self.conv1.weight.device,
                                )
        new_weight = self.conv1.weight.index_select(dim=1, index=mask).clone().detach()
        new_bias = self.conv1.bias.clone().detach()

        new_conv1.weight.requires_grad = False
        new_conv1.weight.copy_(new_weight.contiguous())
        new_conv1.weight.requires_grad = True
        new_conv1.bias.requires_grad = False
        new_conv1.bias.copy_(new_bias.contiguous())
        new_conv1.bias.requires_grad = True

        self.conv1 = new_conv1

    def forward(self, x):
        """
        x: [batch, num_heads, from_seq_len, to_seq_len]
        """
        
        seq_len = x.shape[2]
        assert x.shape[2]%self.block_size == 0
        block_num = seq_len//self.block_size

        out = x.view(x.shape[0], self.num_attention_heads,  block_num, self.block_size, block_num, self.block_size).diagonal(dim1=2, dim2=4)
        out = out.permute(0,4,1,2,3).reshape(-1, self.num_attention_heads, self.block_size, self.block_size)
        # out -> shape [batch, diag block, head, from, to]

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        # out = self.conv3(out).squeeze(dim=1)
        out = torch.flatten(out,start_dim=1)
        # out = self.fc(out).view(-1, self.block_num, 1).squeeze(dim=-1)
        out = self.fc(out).view(x.shape[0], block_num, 2)
        return out

"""
mask: [batch, sequence length], answer mask
"""
def compute_skim_mask(mask, num_block, block_size):
    mask[mask==2] = 0
    blocked_answer_mask = mask.view((-1, num_block, block_size))
    if blocked_answer_mask.shape[0]==1:
        blocked_answer_mask = blocked_answer_mask.squeeze(axis=0)
    skim_label = (torch.sum(blocked_answer_mask, dim=-1)>1).to(dtype=torch.long)
    return skim_label

def test_BlockSkim():
    class DummyConfig():
        max_seq_length = 512
        num_attention_heads = 12
        block_size = 32

    config = DummyConfig()

    block_skim_module = BlockSkim(config)

    x = torch.rand((8, config.num_attention_heads, config.max_seq_length, config.max_seq_length))

    print(block_skim_module(x).shape)

def test_prune_head():

    class DummyConfig():
        max_seq_length = 512
        num_attention_heads = 12
        block_size = 32

    config = DummyConfig()

    block_skim_module = BlockSkim(config)

    x = torch.rand((8, config.num_attention_heads, config.max_seq_length, config.max_seq_length))
    print(block_skim_module(x).shape)

    block_skim_module.prune_heads([0,2,5]) 

    new_x = torch.rand((8, config.num_attention_heads-3, config.max_seq_length, config.max_seq_length))
    print(block_skim_module(new_x).shape)



if __name__ == "__main__":
    # test_BlockSkim()
    test_prune_head()
