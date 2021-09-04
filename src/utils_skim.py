import torch                                        

def skim_hidden_state_tensor(hidden_states, skim_mask_prediction, block_size):
    """
    performance input sequence dim reduction on hidden states tensor
    hidden_states: [batch_size, input_len, hidden_size]
    skim_mask: [batch_size, index]
    """
    hidden_states = hidden_states.view(hidden_states.shape[0], -1, self.block_size, hidden_states.shape[2])
    hidden_states = hidden_states.gather(dim=1, index=skim_mask_prediction.view(*skim_mask_prediction.shape,1,1).expand(-1,-1,block_size,hidden_states.shape[-1]))
    hidden_states = hidden_states.view(hidden_states.shape[0],-1,hidden_states.shape[-1])
    return hidden_states

def skim_attention_mask_tensor(attention_mask, skim_mask_prediction):
    origin_shape = attention_mask.shape
    attention_mask = attention_mask.view(*attention_mask.shape[:3], -1, self.block_size)
    attention_mask = attention_mask.gather(dim=3, index=skim_mask_prediction.view(skim_mask_prediction.shape[0],1,1,skim_mask_prediction.shape[1],1).expand(-1,attention_mask.shape[1],attention_mask.shape[2],-1,self.block_size))
    attention_mask = attention_mask.view(*origin_shape)

def convert_block_mask_to_token(mask, block_size):
    mask = mask.unsqueeze(-1).repeat(1,1,block_size).view(mask.shape[0],-1)
    return mask

def compute_skim_prediction_aligned(skim_mask, block_size, thold=0.5):
    """
    compute the skim prediction for a batch of input to align
    skim_indices [batch, sequence_length]
        False for skimming
        True for remained
    """
    skim_mask = skim_mask.softmax(dim=-1)
    
    # skim decision with threshold 0.5
    # skim_mask_prediction = skim_mask.argmax(dim=-1)
    skim_mask_prediction = (skim_mask[:,:,1]>=thold)

    # calculate input with most tokens remained
    max_remained_length = skim_mask_prediction.sum(dim=-1).max()

    # calculate mask with same tokens remained
    _, skim_indices = skim_mask[:,:,1].topk(dim=1,k=max_remained_length)

    aligned_skim_mask = torch.zeros_like(skim_mask_prediction)
    aligned_skim_mask = aligned_skim_mask.scatter(dim=1, index=skim_indices, value=1)

    aligned_skim_mask = convert_block_mask_to_token(aligned_skim_mask,block_size)

    return aligned_skim_mask.to(dtype=torch.bool)

def trunc_with_mask_batched(input, mask, dim):
    """
    trunc a batched input at dim
        e.g. hidden_states ([batch, seq_len, hidden_size])
            attention_mask ([batch, layer, head, seq_len])
    mask: [batch, seq_len]
    """
    assert input.shape[dim]==mask.shape[1]

    if dim != 1:
        input = input.transpose(1, dim)

    transpose_shape = list(input.shape)
    transpose_shape[1] = -1

    trunc_input = input[mask].view(transpose_shape)

    if dim != 1:
        trunc_input = trunc_input.transpose(1, dim)

    return trunc_input