# DCFSVF
Structural Variation Filter Based on Dual-modal Cross-attention Fusion in Third-generation Sequencing Data

# Installation

## Requirements
- python=3.9.1
- torch=2.5
- pandas=2.2.3
- pysam=0.23.0
- torchvision=0.20.0
- cuda=12.4

## 1. Create a virtual environment

```
#create
conda create -n your_environment_name python=3.9

#activate
conda activate your_environment_name
```

## 2. Package installation

```
pip install -r requirements.txt
```

# Train dataset

```
[https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/Baylor_NGMLR_bam_GRCh37/HG002_PB_70x_RG_HP10XtrioRTG.bam](https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/analysis/NIST_SVs_Integration_v0.6/HG002_SVs_Tier1_v0.6.vcf.gz.)
```

# Usage

python train_main.py --bases_root 01.bases_images --cigar_root 02.cigar_images --class_dirs Del_positive Ins_positive Match_negative  --save_path ./best_model_loss.pth --gpu 0 1 2 --epochs 50 --batch_size 64 --lr 1e-4 --num_workers 4 --train_chrs chr_1 chr_2 chr_3 chr_4 chr_5 chr_6 chr_7 chr_8 chr_9 chr_10 chr_11 chr_12 --test_chrs chr_13 chr_14 chr_15 chr_16 chr_17 chr_18 chr_19 chr_20 chr_21 chr_22 chr_X chr_Y
