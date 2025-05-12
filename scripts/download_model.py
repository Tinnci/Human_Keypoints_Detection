import requests
import os

def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

if __name__ == "__main__":
    model_url = "https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth"
    model_path = "checkpoint_iter_370000.pth"
    
    print("开始下载预训练模型...")
    download_file(model_url, model_path)
    print(f"模型已下载到: {model_path}") 