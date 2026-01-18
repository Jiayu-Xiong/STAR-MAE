from models.modeling_pretrain import *
from datasets.AudioSet import *
from utils.mask_generator import *

train_loader, test_loader = get_dataset_2M(root='path', batch_size=128, num_workers=24)
model = pretrain_videomae_giant_patch14_224()