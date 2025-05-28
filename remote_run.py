from modal import App,Image,gpu,Volume,Image
import os

model_name = "switch_transformer"
app = App(f'PreDiff Sample Run')

image = (
    Image.micromamba(python_version='3.10.12')
    .apt_install("awscli")
    .pip_install_from_requirements(
        requirements_txt='requirements.txt'
    )
    .add_local_dir(
        local_path=os.path.abspath('./'),
        remote_path='/root',
        copy=True,
        ignore= [
            '__pycache__/*',
            './.venv/*',
            './data/*',
            './pretrained_weights/*'
        ]
    )
)

@app.function(
    image=image,
    gpu = 'T4',
    timeout = 86400,
    retries = 0,
    volumes = {
        "/root/sevir_data": Volume.from_name("prediff_vil_precipitation_data"),
        "/root/pretrained_weights": Volume.from_name("prediff_pretrained_weights"),
        "/root/logs": Volume.from_name('prediff_logs')
    }
)
def entry():
    import os
    # os.system('pip freeze > /root/logs/requirements.txt')
    os.system('python -m scripts.train_alignment.train_sevirlr_avg_x --pretrained --cfg /root/scripts/train_alignment/cfg.yaml')
    # import torch

    # if torch.cuda.is_available():
    #     device_count = torch.cuda.device_count()
    #     print(f"Number of available GPUs: {device_count}")
    #     for i in range(device_count):
    #         gpu_name = torch.cuda.get_device_name(i)
    #         print(f"GPU {i}: {gpu_name}")
    # else:
    #     print("No GPU available.")