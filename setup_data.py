import gdown
import os

os.makedirs("data", exist_ok=True)
file_id = "1fGZGWALwTI0Auxzxs5lubuQqlBppxJBP"  # Replace with your actual file ID
output = "data/EVERY_LISTING.gz"
gdown.download("https://drive.google.com/uc?id={file_id}", output, quiet=False)