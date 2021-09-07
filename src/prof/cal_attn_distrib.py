import torch
import numpy as np

def cal_attn_distrib(attn_mat, token_type_ids, ans_mask, p_mask):
    """
    attn_mat: [batch_size, num_head, from_seq_length, to_seq_length]
    """
    all_junk_attn, all_ans_attn = [], []
    for batch_idx in range(attn_mat.shape[0]):
        context_mask = token_type_ids[batch_idx]
        context_mask[p_mask[batch_idx]==1] = 0
        context_mask = context_mask.to(dtype=torch.bool)


        context_attn_mat = attn_mat[batch_idx][:,context_mask][:,:,context_mask]
        
        context_attn_mat = torch.mean(context_attn_mat, dim=1)

        context_ans_mask = ans_mask[batch_idx][context_mask]

        junk_attn = torch.mean(context_attn_mat[:,context_ans_mask==0],dim=1).detach().cpu().numpy()
        ans_attn = torch.mean(context_attn_mat[:,context_ans_mask==1],dim=1).detach().cpu().numpy()

        all_junk_attn.append(junk_attn)
        all_ans_attn.append(ans_attn)

    return all_junk_attn, all_ans_attn

def test_cal_attn_distrib():
    attn_mat = torch.rand((16,12,512,512), device='cuda')
    token_type_ids = torch.randint(0,2,(16,512))
    ans_mask = torch.randint(0,2,(16,512))
    junk_attn, ans_attn = cal_attn_distrib(attn_mat, token_type_ids,ans_mask)


def post_analysis():
    pass


if __name__ == "__main__":
    pass