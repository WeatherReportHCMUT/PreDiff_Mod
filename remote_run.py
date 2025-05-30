from modal import App,Image,gpu,Volume,Image
import os

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
    gpu = 'A100',
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
    os.system('python -m scripts.train_diffusion.train_sevirlr_prediff --cfg ./scripts/train_diffusion/cfg.yaml')