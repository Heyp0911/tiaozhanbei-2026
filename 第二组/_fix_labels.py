"""
_fix_labels.py — 将数据集中的segment标注转为YOLO bbox格式
解决: "Box and segment counts should be equal" 的WARNING
用法: python _fix_labels.py [数据集目录]
"""
import sys, os, shutil
from pathlib import Path

def fix_labels(data_dir="data/ceramic_tiles"):
    """递归遍历labels目录，将segment标注转为bbox，原地修改"""
    data_path = Path(data_dir)
    # 找所有labels目录
    for label_dir in data_path.rglob("labels"):
        fixed_count = 0
        total_files = 0
        for txt_file in label_dir.glob("*.txt"):
            total_files += 1
            lines = txt_file.read_text().strip().split("\n")
            new_lines = []
            modified = False

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                n = len(parts)
                if n == 5:
                    # 标准YOLO bbox: class_id cx cy w h → 保持不变
                    new_lines.append(line.strip())
                elif n > 5:
                    # Segment格式: class_id x1 y1 x2 y2 ... xn yn → 转为bbox
                    cls_id = parts[0]
                    coords = [float(x) for x in parts[1:]]
                    xs = coords[0::2]  # 所有x坐标
                    ys = coords[1::2]  # 所有y坐标
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    modified = True
                    fixed_count += 1

            if modified:
                txt_file.write_text("\n".join(new_lines), encoding="utf-8")

        img_dir = str(label_dir).replace("labels", "images")
        img_count = len(list(Path(img_dir).glob("*"))) if os.path.exists(img_dir) else 0
        print(f"  {label_dir}: {total_files}个标签文件, 修复{fixed_count}个segment→bbox (图片:{img_count}张)")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ceramic_tiles"
    print(f"修复数据集标注: {data_dir}")
    fix_labels(data_dir)
    print("完成！重新运行 _train_ceramic.py 即可。")
