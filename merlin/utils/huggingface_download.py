import os

from huggingface_hub import hf_hub_download


def download_file(
    repo_id: str,      #仓库 ID（文件在网上的哪个仓库里）
    filename: str,     #文件名
    local_dir: str,    #下载到的本地目录
):
    os.makedirs(local_dir, exist_ok=True)
    #local_file_path：local_dir + filename
    local_file_path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir
    )    # Hugging Face存在缓存机制：保证不会重复下载
    print(f"{filename} downloaded and saved to {local_file_path}")
    return local_file_path
