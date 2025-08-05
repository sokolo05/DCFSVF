#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path

def classify_and_copy_images(src_dir: str, dst_root: str):
    """
    根据文件名关键字把图片复制到对应子文件夹。
    src_dir: 原始图片所在目录
    dst_root: 目标根目录，脚本会在其下创建 Ins_positive、Del_positive、Match_negative
    """
    src_path = Path(src_dir)
    dst_root_path = Path(dst_root)

    # 确保目标根目录存在
    dst_root_path.mkdir(parents=True, exist_ok=True)

    # 定义三个子文件夹的 Path 对象
    ins_dir = dst_root_path / "Ins_positive"
    del_dir = dst_root_path / "Del_positive"
    match_dir = dst_root_path / "Match_negative"

    # 创建子文件夹（如果不存在）
    ins_dir.mkdir(exist_ok=True)
    del_dir.mkdir(exist_ok=True)
    match_dir.mkdir(exist_ok=True)

    # 遍历源目录下所有 .png 文件
    for file in src_path.glob("*.png"):
        if "INS" in file.name:
            shutil.copy2(file, ins_dir / file.name)
        elif "DEL" in file.name:
            shutil.copy2(file, del_dir / file.name)
        elif "MATCH" in file.name:
            shutil.copy2(file, match_dir / file.name)
        else:
            # 如果不包含上述关键字，可打印提示或跳过
            print(f"跳过未知类型文件: {file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据文件名关键字（INS/DEL/MATCH）把图片复制到对应文件夹"
    )
    parser.add_argument(
        "source_dir",
        help="原始图片所在目录"
    )
    parser.add_argument(
        "target_root",
        help="目标根目录，将在其下创建 Ins_positive、Del_positive、Match_negative"
    )

    args = parser.parse_args()

    classify_and_copy_images(args.source_dir, args.target_root)
    print("分类与复制完成！")