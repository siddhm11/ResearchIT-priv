"""Fix CRLF line endings in LightGBM model file (Windows compatibility)."""
import os

model_path = os.path.join("models", "reranker-phase6", "production_model", "reranker_v1.txt")

with open(model_path, "rb") as f:
    data = f.read()

crlf_count = data.count(b"\r\n")
print(f"Original size: {len(data)} bytes, CRLF count: {crlf_count}")

if crlf_count > 0:
    data_fixed = data.replace(b"\r\n", b"\n")
    with open(model_path, "wb") as f:
        f.write(data_fixed)
    print(f"Fixed size: {len(data_fixed)} bytes")
    print("Converted CRLF -> LF")
else:
    print("No CRLF found, file is already LF-only")

# Verify
import lightgbm as lgb
model = lgb.Booster(model_file=model_path)
print(f"Model loaded OK: {model.num_trees()} trees, {model.num_feature()} features")
