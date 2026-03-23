import subprocess
from datetime import datetime
from pathlib import Path
import shutil


STEPS = [
    # "src/local/code/crawl_index_data.py",
    # "src/local/code/crawl_stock_data.py",
    "src/local/code/merge_stock_index.py",
]

PROCESSED_DIR = Path("src/local/data/processed")


for step in STEPS:
    subprocess.run(["uv", "run", step], check=True)


archive_base = Path(datetime.now().strftime("%Y%m%d"))
archive_path = shutil.make_archive(str(archive_base), "gztar", root_dir=PROCESSED_DIR)
print(f"打包完成: {archive_path}")
