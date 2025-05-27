import gdown
import os

os.makedirs("data", exist_ok=True)
file_id = "1fGZGWALwTI0Auxzxs5lubuQqlBppxJBP"  # Your actual file ID
output = "data/EVERY_LISTING.gz"

# Use an f-string to interpolate file_id
url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, output, quiet=False)
