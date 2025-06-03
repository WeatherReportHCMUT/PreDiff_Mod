import torch
from lightning.pytorch import Trainer, seed_everything

import warnings
from collections import OrderedDict
from omegaconf import OmegaConf
import os
import argparse

from .alignment_lightning_module import SEVIRAlignmentPLModule
from utils.pl_checkpoint import pl_load
from utils.path import default_pretrained_alignment_dir,pretrained_sevirlr_alignment_name

from dotenv import load_dotenv
_ = load_dotenv('./.env')

pytorch_state_dict_name = "sevirlr_alignment_avgx.pt"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_sevirlr_avg_x', type=str)
    parser.add_argument('--nodes', default=1, type=int,
                        help="Number of nodes in DDP training.")
    parser.add_argument('--gpus', default=1, type=int,
                        help="Number of GPUS per node in DDP training.")
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on SEVIR-LR.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained checkpoints for test.')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.pretrained:
        args.cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "cfg.yaml"))
        assert os.path.exists(os.path.join(default_pretrained_alignment_dir,pretrained_sevirlr_alignment_name)), "Pretrained weights for Knowledge Alignment Model does not exist"
    
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_cfg = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
        float32_matmul_precision = oc_from_file.optim.float32_matmul_precision
    else:
        dataset_cfg = OmegaConf.to_object(SEVIRAlignmentPLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.nodes * args.gpus)
        max_epochs = None
        seed = 0
        float32_matmul_precision = "high"
    torch.set_float32_matmul_precision(float32_matmul_precision)
    seed_everything(seed, workers=True)
    dm = SEVIRAlignmentPLModule.get_sevir_datamodule(
        dataset_cfg=dataset_cfg,
        micro_batch_size=micro_batch_size,
        num_workers=8
    )
    dm.prepare_data()
    dm.setup()
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.nodes * args.gpus)
    total_num_steps = SEVIRAlignmentPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = SEVIRAlignmentPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg
    )
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        num_nodes=args.nodes,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)
    if args.pretrained:
        alignment_ckpt_path = os.path.join(default_pretrained_alignment_dir,pretrained_sevirlr_alignment_name)
        state_dict = torch.load(alignment_ckpt_path,map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict)
        trainer.test(
            model=pl_module,
            datamodule=dm
        )
    elif args.test:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        else:
            ckpt_path = None
        trainer.test(
            model=pl_module,
            datamodule=dm,
            ckpt_path=ckpt_path
        )
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
        trainer.fit(
            model=pl_module,
            datamodule=dm,
            ckpt_path=ckpt_path
        )
        # save state_dict of the latent diffusion model, i.e., EarthformerDiffusion
        pl_ckpt = pl_load(
            path_or_url=trainer.checkpoint_callback.best_model_path,
            map_location=torch.device("cpu")
        )
        # pl_state_dict = pl_ckpt["state_dict"]  # pl 1.x
        pl_state_dict = pl_ckpt
        model_kay = "torch_nn_module."
        state_dict = OrderedDict()
        unexpected_dict = OrderedDict()
        for key, val in pl_state_dict.items():
            if key.startswith(model_kay):
                state_dict[key.replace(model_kay, "")] = val
            else:
                unexpected_dict[key] = val
        torch.save(state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        # test
        trainer.test(
            ckpt_path="best",
            datamodule=dm
        )

if __name__ == "__main__":
    main()
