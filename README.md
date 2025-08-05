# DCFSVF
Structural Variation Filter Based on Dual-modal Cross-attention Fusion in Third-generation Sequencing Data

# Using:
python train_main.py --bases_root /home/laicx/03.study/06.sv_filter/04.bibm/02.split_data/01.bases_images --cigar_root /home/laicx/03.study/06.sv_filter/04.bibm/02.split_data/02.cigar_images --class_dirs Del_positive Ins_positive Match_negative  --save_path ./04.train_loss_deepseek.pth --gpu 0 1 2 --epochs 50 --batch_size 64 --lr 1e-4 --num_workers 4 --train_chrs chr_1 chr_2 chr_3 chr_4 chr_5 chr_6 chr_7 chr_8 chr_9 chr_10 chr_11 chr_12 --test_chrs chr_13 chr_14 chr_15 chr_16 chr_17 chr_18 chr_19 chr_20 chr_21 chr_22 chr_X chr_Y
