import torch

def get_sample_align_fn(sample_align_model):
    r"""
    Code is adapted from https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/scripts/classifier_sample.py#L54-L61
    """
    def sample_align_fn(x, *args, **kwargs):
        r"""
        Calculates `grad(log(p(y|x)))`
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).

        Parameters
        ----------
        x:  torch.Tensor

        Returns
        -------
        grad
        """
        # with torch.inference_mode(False):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = sample_align_model(x_in, *args, **kwargs)
            grad = torch.autograd.grad(logits.sum(), x_in, allow_unused=True)[0]
            return grad
    return sample_align_fn

def get_alignment_kwargs_avg_x(context_seq=None, target_seq=None, ):
    r"""
    Please customize this function for generating knowledge "avg_x_gt"
    that guides the inference.
    E.g., this function uses 2.0 ground-truth future average intensity as "avg_x_gt" for demonstration.

    Parameters
    ----------
    context_seq:    torch.Tensor, aka "y"
    target_seq:     torch.Tensor, aka "x"

    Returns
    -------
    alignment_kwargs:   Dict
    """
    multiplier = 2.0
    batch_size = target_seq.shape[0]
    ret = torch.mean(target_seq.view(batch_size, -1),
                     dim=1, keepdim=True) * multiplier
    return {"avg_x_gt": ret}