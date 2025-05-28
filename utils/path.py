import os 

# region Filename
pretrained_sevirlr_vae_name = 'pretrained_sevirlr_vae_8x8x64_v1_2.pt'
pretrained_sevirlr_alignment_name = 'pretrained_sevirlr_alignment_avg_x_cuboid_v1.pt'
pretrained_sevirlr_earthformer_unet_dir = 'pretrained_sevirlr_earthformerunet_v1.pt'
pretrained_i3d_400_name='pretrained_i3d_400.pt'
pretrained_i3d_600_name='pretrained_i3d_600.pt'
# endregion

# region Path
# TODO: UPdate this if the code move elsewhere
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

default_exps_dir = os.path.abspath(os.path.join(root_dir, "logs"))

default_pretrained_dir = os.path.abspath(os.path.join(root_dir, "pretrained_weights","pretrained_weights"))
default_pretrained_metrics_dir = os.path.abspath(os.path.join(default_pretrained_dir, "metrics"))
default_pretrained_vae_dir = os.path.abspath(os.path.join(default_pretrained_dir, "vae"))
default_pretrained_earthformerunet_dir = os.path.abspath(os.path.join(default_pretrained_dir, "earthformerunet"))
default_pretrained_alignment_dir = os.path.abspath(os.path.join(default_pretrained_dir, "alignment"))
# endregion
if __name__ == "__main__":
    print(root_dir)