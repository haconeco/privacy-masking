# Privacy Masking System

歩行者の顔などの個人情報を検出し、マスキングによる匿名化を行うシステムです。
数万〜数十万枚規模の大量画像処理を前提に、**低コスト・高速・再開可能（Resume-capable）** な設計になっています。

## 特徴
*   **高速・軽量**: Google MediaPipe Face Detector (BlazeFace) を採用。
*   **堅牢なバッチ処理**: マニフェストファイル（CSV）によるステータス管理で、途中停止しても未処理分から再開可能。
*   **スケーラビリティ**: マルチプロセス並列処理に対応。IOバウンド（S3転送）とCPUバウンド（デコード/マスク）を効率化。
*   **柔軟なI/O**: ローカルファイルシステムと Amazon S3 を透過的に扱えます。

## Quick Start (使い方)

### 1. インストール
`uv` を用いて依存関係をセットアップします。

```bash
# プロジェクトのセットアップ
uv sync

# (Optional) MediaPipeモデルのダウンロード
# デフォルトで使用されるモデルを配置済みですが、更新する場合は以下
curl -L https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite -o src/models/face_detection_short_range.tflite
```

### 2. 基本的な実行
実行は `src/main.py` を通して行います。
初回実行時は `--input` と `--output` を指定して**マニフェスト初期化**を行ってください。

#### ローカルファイルを処理する場合
```bash
# マニフェスト(run.csv)を作成し、逐次処理を実行
uv run python src/main.py \
  --manifest run.csv \
  --input ./data/source \
  --output ./data/masked
```

#### S3上の画像を処理する場合（並列処理推奨）
大量の画像を処理する場合は `--parallel` と `--num-processes` を使用して高速化します。

```bash
# 環境変数は事前に設定するか、~/.aws/credentials を使用
export AWS_PROFILE=my-profile

uv run python src/main.py \
  --manifest s3_run.csv \
  --input s3://my-bucket/raw-images/2024/ \
  --output s3://my-bucket/masked-images/2024/ \
  --parallel \
  --num-processes 8
```

### 3. 再開（Resume）とリトライ
中断した場合やエラーが発生した場合、同じマニフェストファイルを指定して再実行すると、**PENDING（未処理）および FAILED（失敗）の画像のみ**を処理します。

```bash
# 同じコマンドを叩くだけでOK
uv run python src/main.py --manifest run.csv ...
```

### 4. GPUの使用（Advanced）
MediaPipeのGPU推論を利用することで、顔検出の高速化が期待できます。
デフォルトはCPU動作ですが、`--device GPU` を指定することでGPUモードに切り替わります。

```bash
uv run python src/main.py \
  --manifest run.csv \
  ... \
  --device GPU
```

> [!IMPORTANT]
> **前提環境について**:
> GPUモードを使用するには、MediaPipeのGPU Delegateがサポートされている環境が必要です。
> - 推奨: **Linux環境 (Ubuntuなど) + NVIDIA GPU driver + CUDA/CuDNN**
> - macOS: 環境によっては動作しない、またはCPUフォールバックが発生する場合があります。
> 
> クラウドインスタンス（AWS g4dn等）で実行する場合に特に有効です。

## アーキテクチャと設計意図

本システムは「大量の画像を、安価な計算機リソースで、確実に処理しきる」ことを目的に設計されています。

### コンポーネント構成 (`src/`)

| ファイル | 役割・設計意図 |
| :--- | :--- |
| **`main.py`** | **エントリーポイント**。<br>CLI引数の解析、マニフェストの初期化、IOのセットアップ、Processorの起動を行います。S3/Localのパス解釈もここで行います。 |
| **`pipeline.py`** | **処理オーケストレーター**。<br>`BatchProcessor` クラスが全体のフローを制御します。<br>- **逐次処理**: デバッグ用。<br>- **並列処理**: `ProcessPoolExecutor` を用い、ワーカープロセスごとにDetectorを独立して初期化することで、Global Interpreter Lock (GIL) を回避しつつ、MediaPipe/TensorFlowの競合を防ぎます。 |
| **`manifest.py`** | **状態管理 (State Management)**。<br>全画像の処理状態 (`PENDING`, `DONE`, `FAILED`) をCSV/Parquetで永続化します。<br>- 数十万行規模でも Pandas で高速に処理可能。<br>-「どこまでやったか」をファイルで管理することで、プロセスがクラッシュしても安全に再開できます。 |
| **`detector.py`** | **顔検出ラッパー**。<br>MediaPipe Face Detector を隠蔽します。<br>- 将来的にモデルを差し替えたり、GPUモードへ切り替えたりする際の変更影響をここに閉じ込めます。 |
| **`masking.py`** | **匿名化ロジック**。<br>- 検出されたBBox（矩形）をそのまま塗るのではなく、**拡張（Expand）**してからぼかすことで、髪の毛や顎などの取り漏れを防ぐ「安全側」の実装になっています。<br>- EXIF情報の除去（OpenCVによる再エンコードで自然に削除）も担当します。 |
| **`io_handler.py`** | **I/O抽象化レイヤー**。<br>`read(uri) -> bytes` / `write(uri, bytes)` というシンプルなインターフェースで、ローカルパスとS3 URI (`s3://`) の違いを吸収します。<br>これにより、ビジネスロジックは「データがどこにあるか」を意識せずに済みます。 |

### ディレクトリ構造

```
.
├── src/
│   ├── main.py       # 実行スクリプト
│   ├── pipeline.py   # パイプライン制御（並列/逐次）
│   ├── manifest.py   # マニフェスト管理
│   ├── detector.py   # 顔検出（MediaPipe）
│   ├── masking.py    # マスキング処理
│   ├── io_handler.py # S3/Local IO
│   ├── config.py     # (Optional) 設定クラス
│   └── models/       # モデルファイル置き場
├── tests/            # テストコード
│   ├── test_manifest.py
│   ├── test_pipeline.py
│   ├── ...
├── README.md         # 本ドキュメント
└── pyproject.toml    # 依存関係定義 (uv/pip)
```

## テスト (Development)

単体テストは `tests/` ディレクトリに集約されています。`pytest` を用いて実行可能です。

```bash
# 全テストの実行
uv run pytest

# 特定のファイルのテスト
uv run pytest tests/test_pipeline.py
```

### テスト範囲
- **Manifest**: CSVの読み書き、ステータス更新ロジック。
- **IO**: Local/S3 (Mock) の読み書き振る舞い。
- **Detector**: ダミー画像に対するBBox返却の形式確認（モデル精度評価ではない）。
- **Masking**: ぼかし処理適用前後の画像変化確認。
- **Pipeline**: モックコンポーネントを用いた正常系・異常系フローの確認。
