import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from wilds import get_dataset
import numpy as np
import os
from tqdm import tqdm
import clip  # 👈 추가된 부분

# ───────────────────────────────────────────────
# 1. 설정 및 장치 확인
# ───────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
# 임베딩을 저장할 최상위 디렉토리
# config.py의 _base_embed_dir과 동일하게 설정할 것
BASE_EMBED_DIR = './data/embeddings' # 경로를 로컬 환경에 맞게 적절히 수정하세요
 
 
# ───────────────────────────────────────────────
# 2. 인코더별 모델 & Transform 정의
# ───────────────────────────────────────────────
 
def get_resnet18():
    """ResNet-18 (ImageNet pretrained) → 512차원"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval().to(device)
 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
    ])
    return model, transform
 
 
def get_clip_rn50():
    """
    CLIP ResNet-50 (OpenAI pretrained) → 1024차원
    공식 clip 패키지를 사용하여 로드
    """
    # model과 CLIP 전용 전처리(preprocess) 함수를 함께 로드
    model, preprocess = clip.load("RN50", device=device)
    
    # 특징 추출을 위한 visual 부분만 가져오고, 
    # 기존 코드의 x.float()와 충돌하지 않도록 fp32로 명시적 변환
    visual = model.visual.float()
    visual.eval().to(device)
 
    return visual, preprocess
 
 
def get_vit_b():
    """ViT-B/16 (ImageNet pretrained) → 768차원"""
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = nn.Identity()
    model.eval().to(device)
 
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
    ])
    return model, transform
 
 
ENCODERS = {
    'resnet18':  get_resnet18,   # 512차원
    'clip_rn50': get_clip_rn50,  # 1024차원
    'vit_b':     get_vit_b,      # 768차원
}
 
 
# ───────────────────────────────────────────────
# 3. Group-balanced val 세트 샘플링
# ───────────────────────────────────────────────
 
def group_balance(feats, targets, bias):
    """
    (target, bias) 조합으로 그룹을 나누고,
    가장 적은 그룹 수에 맞춰 모든 그룹을 동일하게 샘플링하여 반환.
    """
    groups = {}
    for t in np.unique(targets):
        for b in np.unique(bias):
            idx = np.where((targets == t) & (bias == b))[0]
            if len(idx) > 0:
                groups[(t, b)] = idx
 
    min_count = min(len(idx) for idx in groups.values())
    print(f"  [group_balance] 그룹별 샘플 수: "
          f"{ {k: len(v) for k, v in groups.items()} }")
    print(f"  [group_balance] 그룹당 샘플링 수: {min_count}")
 
    balanced_idx = np.concatenate([
        np.random.choice(idx, size=min_count, replace=False)
        for idx in groups.values()
    ])
    np.random.shuffle(balanced_idx)
 
    return feats[balanced_idx], targets[balanced_idx], bias[balanced_idx]
 
 
# ───────────────────────────────────────────────
# 4. 특징 추출
# ───────────────────────────────────────────────
 
def extract_split(model, encoder_name, dataset, split, transform):
    """단일 split에 대해 특징을 추출하여 numpy 배열로 반환."""
    subset = dataset.get_subset(split, transform=transform)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=128, shuffle=False, num_workers=4
    )
 
    bias_field = 'background' if dataset.dataset_name == 'waterbirds' else 'male'
    bias_idx = dataset.metadata_fields.index(bias_field)
 
    all_feats, all_targets, all_bias = [], [], []
 
    with torch.no_grad():
        for x, y_true, metadata in tqdm(loader, desc=f"  {split}"):
            x = x.to(device)
 
            if encoder_name == 'clip_rn50':
                feats = model(x.float()).cpu().numpy()
            else:
                feats = model(x).cpu().numpy()
 
            all_feats.append(feats)
            all_targets.append(y_true.numpy())
            all_bias.append(metadata[:, bias_idx].numpy())
 
    return (
        np.concatenate(all_feats,   axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_bias,    axis=0),
    )
 
 
def save_split(save_dir, split_name, feats, targets, bias):
    """추출된 특징을 config 경로 규칙에 맞게 저장."""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'{split_name}_feats.npy'),   feats)
    np.save(os.path.join(save_dir, f'{split_name}_targets.npy'), targets)
    np.save(os.path.join(save_dir, f'{split_name}_bias.npy'),    bias)
    print(f"  -> '{save_dir}/{split_name}_*.npy' 저장 완료! shape: {feats.shape}")
 
 
# ───────────────────────────────────────────────
# 5. 메인 추출 함수
# ───────────────────────────────────────────────
 
def extract_and_save(dataset_name, encoder_name, root_dir='./data', seed=42):
    print(f"\n{'='*60}")
    print(f"[{dataset_name} / {encoder_name}] 시작")
    print(f"{'='*60}")
 
    np.random.seed(seed)
    model, transform = ENCODERS[encoder_name]()
    dataset = get_dataset(dataset=dataset_name, root_dir=root_dir, download=False)
 
    # config.py의 경로 구조와 일치:
    traintest_dir = os.path.join(
        BASE_EMBED_DIR, f'{dataset_name}_{encoder_name}_embeddings'
    )
    val_dir = os.path.join(
        BASE_EMBED_DIR, f'{dataset_name}_{encoder_name}_val_balanced_embeddings'
    )
 
    # train
    print(f"\n[train] 추출 중...")
    train_feats, train_targets, train_bias = extract_split(
        model, encoder_name, dataset, 'train', transform
    )
    save_split(traintest_dir, 'train', train_feats, train_targets, train_bias)
 
    # val
    print(f"\n[val] 추출 및 group-balancing 중...")
    val_feats, val_targets, val_bias = extract_split(
        model, encoder_name, dataset, 'val', transform
    )
    val_feats_bal, val_targets_bal, val_bias_bal = group_balance(
        val_feats, val_targets, val_bias
    )
    save_split(val_dir, 'val', val_feats_bal, val_targets_bal, val_bias_bal)
 
    # test
    print(f"\n[test] 추출 중...")
    test_feats, test_targets, test_bias = extract_split(
        model, encoder_name, dataset, 'test', transform
    )
    save_split(traintest_dir, 'test', test_feats, test_targets, test_bias)
 
    print(f"\n[{dataset_name} / {encoder_name}] 완료!")
    print(f"  train/test → {traintest_dir}")
    print(f"  val (balanced) → {val_dir}")
 
 
# ───────────────────────────────────────────────
# 6. 실행
# ───────────────────────────────────────────────
 
if __name__ == '__main__':
    DATASETS = ['waterbirds', 'celebA']
 
    TARGET_ENCODERS = ['resnet18']
 
    for dataset in DATASETS:
        for encoder in TARGET_ENCODERS:
            extract_and_save(dataset, encoder, root_dir='./data')
 
    print("\n모든 임베딩 추출 완료!")