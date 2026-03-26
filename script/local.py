import shutil
import subprocess
from datetime import datetime
from pathlib import Path

STEPS = [
    # "src/local/crawl_index_data.py",
    "src/local/financial&candle_data_v0.3.py",
    "src/local/preprocessing_data_v0.6_20260212185825.py",
]

PROCESSED_DIR = Path("data/processed")


for step in STEPS:
    subprocess.run(["uv", "run", step], check=True)


archive_base = Path(datetime.now().strftime("%Y%m%d"))
archive_path = shutil.make_archive(str(archive_base), "gztar", root_dir=PROCESSED_DIR)
print(f"打包完成: {archive_path}")
