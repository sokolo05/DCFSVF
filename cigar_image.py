import argparse
import pandas as pd
import os
import numpy as np
import pysam
import matplotlib.pyplot as plt

# === CIGAR 操作映射 ===
op_to_char = {0: 'M', 1: 'I', 2: 'D', 3: 'N', 4: 'S', 5: 'H', 6: 'P', 7: '=', 8: 'X'}
char_to_num = {
    'M': 1, '=': 1, 'X': 1, '-': 1,  # 匹配类操作，灰色
    'I': 2,                          # 插入，红色
    'D': 3, 'N': 3,                  # 缺失类，绿色
    'S': 5                           # soft clip，蓝色
}

# === 从 BAM 提取编码序列 ===
def process_bam_file(bam_file, chrom, pos, extend_length):

    samfile = pysam.AlignmentFile(bam_file, "rb")
    read_cigar_lists = []
    reconstructed_cigar_lists = []

    start = pos - extend_length
    end = pos + extend_length

    for read in samfile.fetch(chrom, start, end):

        # read not including D and N operations
        read_cigar_seq = []
        for op, length in read.cigartuples:
            if op in (0, 1, 4, 7, 8):  # 保留 M/I/S/=X，不包括 H/P
                op_char = op_to_char.get(op, '?')
                read_cigar_seq.extend([op_char] * length)

        ref_pos = read.reference_start
        pos_in_cigar = None
        idx = 0
        for op, length in read.cigartuples:
            if op in (0, 7, 8):  # M, =, X
                if ref_pos <= pos < ref_pos + length:
                    pos_in_cigar = idx + (pos - ref_pos)
                    break
                ref_pos += length
                idx += length
            elif op in (2, 3):  # D, N
                ref_pos += length
            elif op in (1, 4):  # I, S
                idx += length
            else:
                idx += length
        if pos_in_cigar is None:
            continue
        left = max(0, pos_in_cigar - extend_length)
        right = min(len(read_cigar_seq), pos_in_cigar + extend_length)
        subseq = read_cigar_seq[left:right]
        pad_left = max(0, extend_length - (pos_in_cigar - left))
        pad_right = max(0, extend_length - (right - pos_in_cigar))
        subseq = ['-'] * pad_left + subseq + ['-'] * pad_right
        read_cigar_lists.append(subseq)

        # read including D and N operations
        reconstructed_seq = []
        for op, length in read.cigartuples:
            if op in (0, 1, 2, 3, 4, 7, 8):  # 仍不包括 H, P
                op_char = op_to_char.get(op, '?')
                reconstructed_seq.extend([op_char] * length)

        ref_pos = read.reference_start
        pos_in_cigar = None
        idx = 0
        for op, length in read.cigartuples:
            if op in (0, 7, 8):
                if ref_pos <= pos < ref_pos + length:
                    pos_in_cigar = idx + (pos - ref_pos)
                    break
                ref_pos += length
                idx += length
            elif op in (2, 3):  # D/N
                ref_pos += length
                idx += length
            elif op in (1, 4):  # I/S
                idx += length
            else:
                idx += length
        if pos_in_cigar is None:
            continue
        left = max(0, pos_in_cigar - extend_length)
        right = min(len(reconstructed_seq), pos_in_cigar + extend_length)
        subseq = reconstructed_seq[left:right]
        pad_left = max(0, extend_length - (pos_in_cigar - left))
        pad_right = max(0, extend_length - (right - pos_in_cigar))
        subseq = ['-'] * pad_left + subseq + ['-'] * pad_right
        reconstructed_cigar_lists.append(subseq)

    samfile.close()

    max_len = max(
        max(len(s) for s in read_cigar_lists) if read_cigar_lists else 0,
        max(len(s) for s in reconstructed_cigar_lists) if reconstructed_cigar_lists else 0
    )
    for s in read_cigar_lists + reconstructed_cigar_lists:
        if len(s) < max_len:
            s.extend(['-'] * (max_len - len(s)))

    return read_cigar_lists, reconstructed_cigar_lists

# === 绘图函数 ===
def plot_cigar_matrix(cigar_list, extend_length, output_path):
    num_matrix = np.array([[char_to_num.get(c, 0) for c in row] for row in cigar_list])
    rgb_image = np.zeros((*num_matrix.shape, 3))

    cmap = {
        0: [1, 1, 1],         # padding 或未知操作：白色
        1: [0.6, 0.6, 0.6],   # 匹配类（M/=/X/-）：灰色
        2: [1.0, 0.0, 0.0],   # 插入 I：红色
        3: [0.0, 1.0, 0.0],   # 缺失 D/N：绿色
        5: [0.0, 0.0, 1.0],   # soft clip S：蓝色
    }

    for num, color in cmap.items():
        mask = num_matrix == num
        rgb_image[mask] = color

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_image, aspect='auto', interpolation='nearest')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# === 主程序入口 ===
def main():
    parser = argparse.ArgumentParser(description="CIGAR-based SV region visualization from TXT")
    parser.add_argument('--txt_file', required=True, help="TXT file with SV info")
    parser.add_argument('--bam_file', required=True, help="Input BAM file")
    parser.add_argument('--output_dir', required=True, help="Directory to save images")
    parser.add_argument('--extend_length', type=int, default=500, help="Extension length")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.txt_file, sep='\t', header=0)

    print(f"共读取 {len(df)} 条记录")

    for idx, row in df.iterrows():
        chrom = str(row['chr'])
        pos = int(row['pos'])
        svtype = str(row['sv_type'])
        svlen = str(row['sv_len'])

        read_cigar_lists, reconstructed_cigar_lists = process_bam_file(
            args.bam_file, chrom, pos, args.extend_length
        )

        prefix = f"chr_{chrom}_{pos}_{svtype}_{svlen}"
        read_path = os.path.join(args.output_dir, f"{prefix}_query_read.png")
        recon_path = os.path.join(args.output_dir, f"{prefix}_recon_read.png")

        if read_cigar_lists:
            plot_cigar_matrix(read_cigar_lists, args.extend_length, read_path)
            print(f"已保存: {read_path}")
        if reconstructed_cigar_lists:
            plot_cigar_matrix(reconstructed_cigar_lists, args.extend_length, recon_path)
            print(f"已保存: {recon_path}")

    print(f"全部完成，图像保存在: {args.output_dir}")

if __name__ == "__main__":
    main()