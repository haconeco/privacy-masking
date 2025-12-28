# TODO List

- [x] **初期化と環境構築**
    - [x] `uv init` によるプロジェクト初期化
    - [x] 依存ライブラリの追加 (`mediapipe`, `opencv-python`, `boto3`, `pandas`, `pytest` 等)
    - [x] ディレクトリ構造の決定

- [x] **設計 (Planning)**
    - [x] `implementation_plan.md` の作成（コンポーネント詳細設計）

- [x] **実装 (Core Components)**
    - [x] **Manifest Manager**: CSV/Parquetの読み書き、ステータス管理
    - [x] **IO Handler**: ローカル/S3からの画像読み込み・保存クラス
    - [x] **Face Detector**: MediaPipe Face Detector のラッパークラス（GPU/CPU切り替え考慮）
    - [x] **Masking Processor**: 画像へのぼかし・モザイク処理、EXIF除去
    - [x] **Pipeline**: 各コンポーネントを繋ぐ処理フロー（Multiprocessing対応）
        - [x] Sequential Implementation
        - [x] Parallel Implementation (Multiprocessing)

- [x] **性能検証 (Performance Verification)**
    - [x] Parallel pipeline smoke test
