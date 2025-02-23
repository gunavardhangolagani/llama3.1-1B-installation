from huggingface_hub import snapshot_download

snapshot_download(
                    repo_id="meta-llama/Llama-3.2-1B-Instruct",
                    use_auth_token="INSERT YOUR TOKEN HERE",
                    local_dir="C:\\codes\\llama1B\\")
