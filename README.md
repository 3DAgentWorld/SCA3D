<p align="center">
  <h1 align="center"> SCA3D: Enhancing Cross-modal 3D Retrieval via 3D Shape and Caption Paired Data Augmentation
  </h1>
  <p align="center">
    <a ><strong>Junlong Ren*</strong></a>
    ·
    <a ><strong>Hao Wu*</strong></a>
    ·
    <a ><strong>Hui Xiong</strong></a>
    ·
    <a href="https://wanghao.tech//"><strong>Hao Wang✉</strong></a>
  </p>
  <p align="center">The Hong Kong University of Science and Technology (GuangZhou)</p>
  <p align="center">(* Equal Contribution)</p>

  <h3 align="center"> ICRA 2025</h3>

<p align="center">
   <a href="https://arxiv.org/pdf/2502.19128"><img src="https://img.shields.io/badge/arXiv-Paper-red.svg"></a>
</p>
<div align=center><img src=assets\overview.jpg></div>

<div align=center><img src=assets\DataAug.jpg></div>

# Preparation

## Downloading Datasets

The point clouds, text captions, and segmentation annotations are provided by [Parts2Words](https://github.com/JLUtangchuan/Parts2Words). You can download the files [here](https://drive.google.com/file/d/11uSuGUxV7WSM3Cogh4tZmt37ZPYCffVE/view?usp=sharing).

## Checkpoints

We also provide the pre-trained model weights in [Google Drive](https://drive.google.com/file/d/1uh_exEcNpB9uTwSheLkAPI6S5LQmradN/view?usp=sharing).

# Quick Start

```bash
# Train
CUDA_VISIBLE_DEVICES=0 python train.py --config config/SCA3D.yaml

# Eval
CUDA_VISIBLE_DEVICES=0 python val.py --config config/SCA3D.yaml
```

# Citation

If you feel this project helpful to your research, please cite our work.

```bibtex
@article{ren2025sca3d,
  title={SCA3D: Enhancing Cross-modal 3D Retrieval via 3D Shape and Caption Paired Data Augmentation},
  author={Ren, Junlong and Wu, Hao and Xiong, Hui and Wang, Hao},
  journal={arXiv preprint arXiv:2502.19128},
  year={2025}
}
```

# Acknowledgement

- This work is built on [Parts2Words](https://github.com/JLUtangchuan/Parts2Words) and we borrow some codes from [CoVR-ECDE](https://github.com/OmkarThawakar/composed-video-retrieval). Thanks for these great works.