import subprocess
from datetime import datetime
from pathlib import Path
import shutil


STEPS = [
    "src/server/code/roll_generate_index_v0.7_20260111154500.py",
    "src/server/code/process_data_NaNto-1000_20260213.py",
    "src/server/code/extract_company_data_v0.3_20260126155425.py",
    "src/server/code/parquet_to_predata_v0.01_20260214.py",
    "src/server/code/train_timexer.py",
    "src/server/code/predict_and_evaluate_v0.1_20260212123505.py",
]


for step in STEPS:
    subprocess.run(["uv", "run", step], check=True)



