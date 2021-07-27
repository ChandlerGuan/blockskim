import torch
import torch.nn as nn

class BlockSkim(nn.Module):
    def __init__(self, config):
        super(BlockSkim, self).__init__()

        self.kernel_size = 3
        self.stride = 1
        self.seq_len = config.max_seq_length
        self.pool_kernel_size = 2

        self.block_size = config.block_size
        self.block_num = self.seq_len//self.block_size
        self.num_attention_heads = config.num_attention_heads

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

    def forward(self, x):
        out = x.view(-1, self.num_attention_heads,  self.block_num, self.block_size, self.block_num, self.block_size).diagonal(dim1=2, dim2=4)
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
        out = self.fc(out).view(-1, self.block_num, 2)
        return out

"""
mask: [batch, sequence length], answer mask
"""
def compute_skim_mask(mask, num_block, block_size):
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

if __name__ == "__main__":
    test_BlockSkim()
