import torch
import torch.nn.functional as F

"""
The PGD attack function in this file is used for adversarial training.

During adversarial training of OMP model, at each iteration of PGD attack,
    we RANDOMLY select a path to perform attack.

"""

# -------- PGD attack --------
def pgd_attack(net, image, label, eps, alpha=0.01, iters=7, random_start=True, d_min=0, d_max=1):

    perturbed_image = image.clone()
    perturbed_image.requires_grad = True

    image_max = image + eps
    image_min = image - eps
    image_max.clamp_(d_min, d_max)
    image_min.clamp_(d_min, d_max)

    # initialize average batch data gradient sign sum
    avg_batch_grad_sign_sum = 0

    if random_start:
        with torch.no_grad():
            perturbed_image.data = image + perturbed_image.uniform_(-1*eps, eps)
            perturbed_image.data.clamp_(d_min, d_max)
    
    for idx in range(iters):
        net.zero_grad()
        _, logits = net(perturbed_image, 'random')

        loss = F.cross_entropy(logits, label)
        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()
        
        loss.backward()
        data_grad = perturbed_image.grad.data

        # collect batch grad sign sum in each iteration
        avg_batch_grad_sign_sum = avg_batch_grad_sign_sum + torch.sum(torch.abs(torch.sign(data_grad)))

        with torch.no_grad():
            perturbed_image.data += alpha * torch.sign(data_grad)
            perturbed_image.data = torch.max(torch.min(perturbed_image, image_max), image_min)
    perturbed_image.requires_grad = False
    
    # compute average batch grad sign sum
    avg_batch_grad_sign_sum = avg_batch_grad_sign_sum / iters
    
    return perturbed_image, avg_batch_grad_sign_sum

