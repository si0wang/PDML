import torch

def TV_Distance(mu1, sigma1, mu2, sigma2):
    # TV(2||1)
    sigma_diag_1 = torch.diag_embed(sigma1, offset=0, dim1=-2, dim2=-1)
    sigma_diag_2 = torch.diag_embed(sigma2, offset=0, dim1=-2, dim2=-1)

    sigma_diag_2_inv = sigma_diag_2.inverse()

    # log(det(sigma2^T)/det(sigma1))
    term_1 = (sigma_diag_2.det() / sigma_diag_1.det()).log()
    # term_1[term_1.ne(term_1)] = 0

    # trace(inv(sigma2)*sigma1)
    term_2 = torch.diagonal((torch.matmul(sigma_diag_2_inv, sigma_diag_1)), dim1=-2, dim2=-1).sum(-1)

    # (mu2-m1)^T*inv(sigma2)*(mu2-mu1)
    term_3 = torch.matmul(torch.matmul((mu2 - mu1).unsqueeze(-1).transpose(2, 1), sigma_diag_2_inv),
                          (mu2 - mu1).unsqueeze(-1)).flatten()

    # dimension of embedded space (number of mus and sigmas)
    n = mu1.shape[1]

    # Estimate upperbound of TV distance on entire batch
    tv = 0.5 * torch.sqrt(term_1 - n + term_2 + term_3)

    return torch.mean(tv)