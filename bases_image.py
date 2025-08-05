import argparse
import pysam
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# === ACGTN 编码映射 ===
base_to_num = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5, '-': 0}

# === 画图函数 ===
def plot_matrix(seq_list, extend_length, output_path):
    cmap = {
        0: [1, 1, 1],       # '-'
        1: [1, 0, 0],       # A
        2: [0, 1, 0],       # C
        3: [0, 0, 1],       # G
        4: [1, 1, 0],       # T
        5: [0.7, 0.7, 0.7]  # N
    }
    arr = np.array(seq_list)
    rgb_image = np.zeros((*arr.shape, 3))
    for num, color in cmap.items():
        rgb_image[arr == num] = color

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_image, aspect='auto', interpolation='nearest')
    ax.axvline(x=extend_length, color='white', linestyle='--', linewidth=1.0)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# === 主处理函数 ===
def process_record(bam, ref, chrom, pos, svtype, svlen, extend_length):
    samfile = pysam.AlignmentFile(bam, "rb")
    ref_genome = pysam.FastaFile(ref)

    reconstructed_list = []
    query_only_list = []

    start = pos - extend_length
    end = pos + extend_length

    for read in samfile.fetch(chrom, start, end):
        cigar = read.cigartuples
        read_pos = read.reference_start
        query_pos = 0
        query_seq = read.query_sequence

        reconstructed_sequence = []
        pos_in_reconstructed = None

        for op, length in cigar:
            if op in (0, 7, 8):  # M, =, X
                if read_pos <= pos < read_pos + length:
                    pos_in_reconstructed = len(reconstructed_sequence) + (pos - read_pos)
                reconstructed_sequence.extend(ref_genome.fetch(chrom, read_pos, read_pos + length))
                read_pos += length
                query_pos += length
            elif op in (1, 4):  # I, S
                reconstructed_sequence.extend(query_seq[query_pos:query_pos + length])
                query_pos += length
            elif op in (2, 3):  # D, N
                reconstructed_sequence.extend(ref_genome.fetch(chrom, read_pos, read_pos + length))
                read_pos += length

        if pos_in_reconstructed is not None:
            left = max(0, pos_in_reconstructed - extend_length)
            right = min(len(reconstructed_sequence), pos_in_reconstructed + extend_length)
            subseq = reconstructed_sequence[left:right]
            pad_left = max(0, extend_length - (pos_in_reconstructed - left))
            pad_right = max(0, extend_length - (right - pos_in_reconstructed))
            subseq = ['-'] * pad_left + list(subseq) + ['-'] * pad_right
            subseq_num = [base_to_num.get(b.upper(), 0) for b in subseq]
            reconstructed_list.append(subseq_num)

        query_pos_in_read = None
        query_pos = 0
        ref_pos = read.reference_start
        for op, length in cigar:
            if op in (0, 7, 8):
                if ref_pos <= pos < ref_pos + length:
                    query_pos_in_read = query_pos + (pos - ref_pos)
                ref_pos += length
                query_pos += length
            elif op == 1 or op == 4:
                query_pos += length
            elif op == 2 or op == 3:
                ref_pos += length

        if query_pos_in_read is not None:
            read_seq = read.query_sequence
            left = max(0, query_pos_in_read - extend_length)
            right = min(len(read_seq), query_pos_in_read + extend_length)
            subseq = read_seq[left:right]
            pad_left = max(0, extend_length - (query_pos_in_read - left))
            pad_right = max(0, extend_length - (right - query_pos_in_read))
            subseq = '-' * pad_left + subseq + '-' * pad_right
            subseq_num = [base_to_num.get(b.upper(), 0) for b in subseq]
            query_only_list.append(subseq_num)

    samfile.close()
    ref_genome.close()

    max_len = max(
        max(len(s) for s in reconstructed_list) if reconstructed_list else 0,
        max(len(s) for s in query_only_list) if query_only_list else 0
    )
    for s in reconstructed_list + query_only_list:
        if len(s) < max_len:
            s.extend([0] * (max_len - len(s)))

    return query_only_list, reconstructed_list

# === 主程序入口 ===
def main():
    parser = argparse.ArgumentParser(description="Generate SV region visualization from BAM and TXT")
    parser.add_argument('--txt_file', required=True, help="Input TXT file with SV positions")
    parser.add_argument('--bam_file', required=True, help="Aligned reads BAM file")
    parser.add_argument('--ref_genome', required=True, help="Reference genome (FASTA)")
    parser.add_argument('--output_dir', required=True, help="Directory to save images")
    parser.add_argument('--extend_length', type=int, default=500, help="Extension length around SV position")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.txt_file, sep='\t', header=0)

    print(f"共读取 {len(df)} 条记录")

    for idx, row in df.iterrows():
        chrom = str(row['chr'])
        pos = int(row['pos'])
        svtype = str(row['sv_type'])
        svlen = str(row['sv_len'])

        query_only, reconstructed = process_record(
            args.bam_file, args.ref_genome, chrom, pos, svtype, svlen, args.extend_length
        )

        prefix = f"chr_{chrom}_{pos}_{svtype}_{svlen}"
        query_path = os.path.join(args.output_dir, f"{prefix}_query_only.png")
        recon_path = os.path.join(args.output_dir, f"{prefix}_reconstructed.png")

        if query_only:
            plot_matrix(query_only, args.extend_length, query_path)
            print(f"已保存: {query_path}")
        if reconstructed:
            plot_matrix(reconstructed, args.extend_length, recon_path)
            print(f"已保存: {recon_path}")

    print(f"全部完成，图像保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
