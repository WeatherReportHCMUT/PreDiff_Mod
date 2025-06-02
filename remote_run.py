from modal import App,Image,gpu,Volume,Image
import os
from dotenv import load_dotenv
_ = load_dotenv('./.env')

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
    .env({
        "WANDB_API_KEY": os.getenv('WANDB_API_KEY'),
        "WANDB_ENTITY": os.getenv('WANDB_ENTITY'),
        'WANDB_PROJECT':os.getenv('WANDB_PROJECT')
    })
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
    # from dotenv import load_dotenv
    
    os.system('python -m scripts.train_vae.train_vae_sevirlr --cfg ./scripts/train_vae/cfg.yaml')