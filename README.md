# Explanation_Tuning (LLM + LoRA + ZeRO)
LLMのLoRAチューニング

### 作成する必要のあるディレクトリ
- `LLaVa-FT-datikz/results`
```
cd LLaVa-FT-datikz
mkdir results
```

### 🔧 環境変数ファイルの作成

このリポジトリでは `.env` ファイルを使って環境変数を管理しています．

```
cp .env.example .env
```
.env を編集して、必要な値を設定してください。


### Train 
```
bash scripts/train_DDP.sh
```
sbatchでの実行
```
sbatch sbatch_scripts/sbatch_train.sh
```
