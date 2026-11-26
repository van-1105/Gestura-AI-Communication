
import csv
import shutil
from pathlib import Path

path = Path("backend/data/gesture_samples.csv")
backup = Path("backend/data/gesture_samples_backup.csv")
temp = Path("backend/data/gesture_samples_fixed.csv")

# Read the original dataset
with path.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))

if not rows:
    raise SystemExit("Dataset is empty.")

# Validate actual data rows: 1 label + 63 coordinates
bad_rows = [
    i + 1 for i, row in enumerate(rows[1:], start=1)
    if len(row) != 64
]

if bad_rows:
    print("Incorrect data rows:", bad_rows[:10])
    raise SystemExit("Repair stopped. Original file unchanged.")

# Create a correct header
header = ["label"] + [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

# Preserve the original file without overwriting an existing backup
if not backup.exists():
    shutil.copy2(path, backup)

# Replace only the header; preserve every data row
with temp.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows[1:])

temp.replace(path)

print("DATASET REPAIRED SUCCESSFULLY")
print("Total samples:", len(rows) - 1)
print("Columns:", len(header))
print("Backup preserved:", backup.exists())
