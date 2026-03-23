import subprocess
from pathlib import Path
import tarfile

p = Path("data")
p.mkdir(parents=True, exist_ok=True)
if not any(p.iterdir()):
    # 获取当前目录下最新的 tar.gz 文件
    gz_files = list(Path(".").glob("*.tar.gz"))
    tarball = max(gz_files, key=lambda f: f.stat().st_mtime)
    # 解压到 data 目录
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(path=p)

STEPS = [
    # "src/server/roll_generate_index_v0.7_20260111154500.py",
    # "src/server/process_data_NaNto-1000_20260213.py",
    # "src/server/parquet_to_predata_v0.01_20260214.py",
    "src/server/train_timexer.py",
    "src/server/predict_and_evaluate_v0.1_20260212123505.py",
]


for step in STEPS:
    subprocess.run(["uv", "run", step], check=True)
