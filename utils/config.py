# ----------------------Common Hyperparams-------------------------- #
num_class = 1
mlp_neurons = 128
 
# ----------------------Encoder Selection-------------------------- #
# 사용할 인코더를 여기서만 변경하면 hid_dim과 경로가 자동으로 설정됩니다.
# 선택 가능: 'resnet18' | 'clip_rn50' | 'vit_b'
encoder = 'resnet18'
 
_encoder_hid_dim = {
    'resnet18':  512,
    'clip_rn50': 1024,
    'vit_b':     768,
}
 
assert encoder in _encoder_hid_dim, (
    f"지원하지 않는 인코더: '{encoder}'. "
    f"선택 가능: {list(_encoder_hid_dim.keys())}"
)
 
hid_dim = _encoder_hid_dim[encoder]
 
# ----------------------Baseline Hyperparams-------------------------- #
# 이 부분을 수정하여 최적의 Accuracy를 냅니다.
base_epochs = 100
base_batch_size = 256
base_lr = 0.0001
weight_decay = 1  # Vary this to train a bias-amplified model
scale = 8
std = 0.2
K = 6
 
opt_b = 'adam'
opt_m = 'adam'
 
# ----------------------Paths-------------------------- #
# 새로 만들 모델의 학습 자료
basemodel_path = f'./new_models/basemodel_{encoder}.pth'
margin_path    = f'./new_models/margin_{encoder}.pth'

# 기존에 존재했던 모델의 학습 자료
# basemodel_path = './saved_models/basemodel_res.pth' 
# margin_path    = './saved_models/margin_res.pth'
 
# ----------------------Model-details-------------------------- #
model_name = encoder
 
# ----------------------ImageNet Means and Transforms---------- #
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
 
# -----------------------CelebA/Waterbirds-parameters--------- #
dataset_path     = './data'                   # CelebA 데이터 루트
img_dir          = './data/celebA/img_align_celeba' # 이미지 폴더
partition_path   = './data/celebA/list_eval_partition.csv' # 분할 CSV
attr_path        = './data/celebA/list_attr_celeba.csv'    # 속성 CSV
target_attribute = 'Blond_Hair'
bias_attribute   = 'Male'

# 인코더 이름에 따라 임베딩 경로 자동 설정 (상대 경로 적용)
_base_embed_dir = './data/embeddings'

celeba_path          = f'{_base_embed_dir}/celebA_{encoder}_embeddings'
celeba_val_path      = f'{_base_embed_dir}/celebA_{encoder}_val_balanced_embeddings'
waterbirds_path      = f'{_base_embed_dir}/waterbirds_{encoder}_embeddings'
waterbirds_val_path  = f'{_base_embed_dir}/waterbirds_{encoder}_val_balanced_embeddings'