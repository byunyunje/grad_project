# 🎓 졸업 프로젝트: [프로젝트 이름 미정]

이 프로젝트는 **Mitigating biases in blackbox feature extractors for image classification tasks** 논문의 문제점을 인식하고, 문제가 되는 코드를 수정하는 **4인 팀 프로젝트**입니다.

---

## 📌 개요

| 항목 | 내용 |
|------|------|
| **기간** | 2026.03 ~ 2026.12 (2학기) |
| **목표** | 블랙박스 모델의 편향성 완화 및 공정성 및 신뢰성 향상 |
| **팀 구성** | 4인 |

---

## 📚 출처 및 인용 (Reference)

본 프로젝트는 아래 논문의 코드를 기반으로 개선 및 연구를 진행하고 있습니다.

- **원본 논문:** Mitigating biases in blackbox feature extractors for image classification tasks
- **원본 코드:** https://github.com/abhipsabasu/blackbox_bias_mitigation

---

## 🛠️ 개발 환경 구축 가이드 (초기 세팅)

팀원들은 아래 순서대로 환경을 구축해 주세요. **(Windows 10/11 기준)**

### 1. WSL2 및 Ubuntu 설치

Windows에서 리눅스 환경을 사용하기 위해 설치합니다.

1. **터미널(PowerShell)** 을 관리자 권한으로 실행
2. 아래 명령어 입력 후 재부팅
```powershell
wsl --install
```

3. 설치된 **Ubuntu** 를 실행하여 사용자 계정(`ID/PW`) 생성

---

### 2. VS Code 연동

1. [VS Code](https://code.visualstudio.com/) 설치 후 **WSL** 확장 프로그램(Microsoft 제작) 설치
2. Ubuntu 터미널에서 프로젝트 폴더로 이동 후 아래 명령어 입력
```bash
code .
```

---

### 3. Pixi (패키지 매니저) 설치

Python 및 라이브러리 관리를 위해 `pixi`를 사용합니다. Ubuntu 터미널에서 실행하세요.
```bash
curl -fsSL https://pixi.sh/install.sh | bash

# 설치 후 터미널을 껐다 켜거나 아래 명령어 실행
source ~/.bashrc
```

> **참고:**https://gli.konkuk.ac.kr/board/lectures/machine-learning/exercises/environment-setup.html (오병국 교수님 사이트)에서 자세한 사항을 체크해주세요.

---

### 4. GitHub 저장소 클론

Ubuntu 터미널에서 아래 명령어를 순서대로 실행하세요.
```bash
# 1. 프로젝트를 저장할 디렉토리로 이동 (예시)
cd ~

# 2. 저장소 클론
git clone https://github.com/byunyunje/grad_project.git

# 3. 프로젝트 폴더로 이동
cd grad_project
```

> **참고:** 클론 후 `git remote -v` 명령어로 원격 저장소가 정상 연결됐는지 확인할 수 있습니다.

---

## 📁 프로젝트 폴더 구조
```
grad_project/
├── .pixi/                  # Pixi 환경 설정 (자동 생성)
├── data/                   # 데이터셋(이미지, npy)
├── models/                 # 모델 코드
├── new_models/             # 우리가 학습한 모델 저장
├── saved_models/           # 미리 학습된 모델 저장
├── utils/                  # 유틸리티 함수
├── extract_features.py     # 특징 추출 스크립트
├── margin_loss.py          # Margin Loss 구현
├── .gitattributes
├── .gitignore
├── pixi.lock               # Pixi 패키지 잠금 파일
├── pixi.toml               # Pixi 환경 설정
├── requirements.txt        # 패키지 의존성 목록
├── 졸프 테스트 결과.txt     # 테스트 결과 기록
├── log.txt                 # 코드 변경 사항 기록
└── README.md
```

> **참고:** 이미지, npy 파일은 용량 문제로 인해 GItHub에 업로드할 수 없었습니다.
다운로드 링크는 https://drive.google.com/file/d/1OscO8ibNTFov4e7XeHDCHlhfZ4mUy6vs/view?usp=sharing 입니다.

---

## 🚀 실행 방법 (How to Run)
```bash
# 패키지 환경 설치
pixi install

# 학습 실행 (예시)
pixi run python margin_loss.py --dataset waterbirds --train --type baseline --bias --seed (...)
pixi run python margin_loss.py --dataset waterbirds --train --type margin --seed (...)

# 평가 실행 (예시)
pixi run python margin_loss.py --dataset waterbirds --type baseline --clustering
```

> 이외에도 명령어가 많으니 https://github.com/abhipsabasu/blackbox_bias_mitigation 를 참고해주세요.

---