from huggingface_hub import snapshot_download

model_id = "pierreguillou/gpt2-small-portuguese"
local_dir = "./model-gpt2"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("Modelo baixado em:", local_dir)