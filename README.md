# Orthogonal-Multi-Path

This repo provides the PyTorch codes of the paper [*Learn Robust Features via Orthogonal Multi-Path*](https://arxiv.org/abs/2010.12190v1).

Please refer to [*repo*](https://github.com/fanghenshaometeor/Diversity-Via-Orthogonality) for the code of the latest work, [*Towards Robust Neural Networks Via Orthogonal Diversity*](https://arxiv.org/abs/2010.12190).

This repo will no longer get updated.

<!-- # Dependencies
- python 3.6 (miniconda)
- PyTorch 1.5.0

# File Descriptions

- `train.sh,.py` training scripts for OMP model
- `train_ablation.sh,py` ablation training scripts for OMP model
- `test.sh,.py` test scripts for OMP model
- `white_attack_1,2,3.sh,.py` white-box attack scripts for OMP model
- `black_attack.sh,.py` black-box attack scripts for OMP model

# Usage

We provide trained model files in the `./save/` directory. Users could directly check the performance of these models.

## training

To reproduce the training, users can run the `train.sh` shell scripts directly on the command line.
```
sh train.sh
```

## test

To test the performance of each path in an OMP model, users can run the `test.sh` shell scripts directly on the command line.
```
sh test.sh
```

## attack

To evaluate the robustness of OMP model, users can run the attack scripts directly on the command line. Detailed descriptions of every attack script are listed as follows.
- **white_attack_1** performs white-box FGSM and PGD attacks on **EACH** path in an OMP model. In this setting, each path in the OMP model is viewed as a single network, and we evaluate the robustness of these individual networks.
- **white_attack_2** performs white-box FGSM and PGD attacks on the **SELECTED** path in an OMP model. The resulting adversarial examples are then reclassified by **OTHER** paths in the OMP model. In this setting, we evaluate the transferability of the adversarial examples among different paths in an OMP model. 
- **white_attack_3** performs white-box FGSM and PGD attacks on **ALL** the paths in an OMP model. The resulting adversarial examples are then reclassified by **OTHER** paths in the OMP model. In this setting, we evaluate the robustness of each path by simultaneously attacking all the paths.
- **black_attack** performs white-box FGSM and PGD attacks on vanilla-trained networks. The resulting adversarial examples are then reclassified by each path in an OMP model. In this setting, we evaluate the robustness of OMP model against black-box attacks.

## 

If u have problems about the codes or paper, u could contact me (fanghenshao@sjtu.edu.cn) or raise issues in GitHub.

If u find the codes useful, welcome to fork and star this repo and cite our paper! :)

```
@ARTICLE{2020arXiv201012190F,
       author = {{Fang}, Kun and {Wu}, Yingwen and {Li}, Tao and {Huang}, Xiaolin and
         {Yang}, Jie},
        title = "{Learn Robust Features via Orthogonal Multi-Path}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
         year = 2020,
        month = oct,
          eid = {arXiv:2010.12190},
        pages = {arXiv:2010.12190},
archivePrefix = {arXiv},
       eprint = {2010.12190},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201012190F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
``` -->