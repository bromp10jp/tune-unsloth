import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
#from transformers import TrainingArguments

max_seq_length = 2048 # シーケンス長を設定
model_name = vars().get('MODEL', 'unsloth/Llama-3.2-3B-Instruct')
BASE = vars().get('BASE', './')
TUNE_DATA_FILE = vars().get('TUNE_DATA_FILE', 'sample_dataset.jsonl')
DATA_FILE = BASE + TUNE_DATA_FILE
save_dir = BASE + "finetuned_model"

# dtype の選定（CPUは考慮しない） ※新しいGPUだとbfloat16が使える（AWSだとg5やg6系）
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    compute_dtype = torch.bfloat16
    use_bf16 = True
    use_fp16 = False
else:
    compute_dtype = torch.float16
    use_bf16 = False
    use_fp16 = True
print(f"compute_dtype: {compute_dtype} use_bf16: {use_bf16} use_fp16: {use_fp16}")

# 4ビット動的量子化でモデルを読み込みます
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = compute_dtype,
    load_in_4bit = True, # 4ビット量子化を有効化
)

# JSONLの各行が {instruction, context, response}
# 例：{"instruction": "東京観光のお薦めは？","context": "あなたは観光ガイドです", "response": "メジャー処は浅草です。"}
# localファイルなら 'json' で読み込み
print("教師データのロード")
dataset = load_dataset("json", data_files=DATA_FILE, split="train").train_test_split(test_size=0.1, seed=42)
print(dataset)

def formatting_function(example):
    # Llama 3.2-Instructのテンプレートに合わせてメッセージを整形
    messages = [
        {"role": "system", "content": example["context"]},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]}
    ]
    # apply_chat_templateを使い、テキスト形式に変換
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return { "text" : text }

dataset = dataset.map(formatting_function)

# LoRAアダプターの適用（Peftモデルの作成）
model = FastLanguageModel.get_peft_model(
    model,
    r=16,               # LoRAのランク（低い：メモリ節約、高い：性能向上、デフォルトは16）
    lora_alpha=32,      # LoRAのスケーリングファクター（目安はrの2倍）
    lora_dropout=0.05,  # ドロップアウト率（過学習防止、0.0〜0.1程度が一般的）
    #target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    target_modules=["q_proj","v_proj"], # 軽量化のためQueryとValue層のみ。本来はAttentionとfeed-forward両方に適用するのが望ましい
    bias="none",        # バイアスの扱い（"none", "all", "lora_only"から選択）(noneだとバイアス固定で学習されない)
)
# トレーナーの設定（SFTとはSupervised Fine-Tuningの略です）
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],  # 学習用データセット
    eval_dataset=dataset["test"],    # 検証用データセット
    dataset_text_field="text",       # データセット内のテキストフィールド名
    max_seq_length=max_seq_length,
    packing=True,
    args=SFTConfig(
        # バッチサイズ設定
        per_device_train_batch_size = 2, # 各デバイスのバッチサイズ
        gradient_accumulation_steps = 8, # 勾配蓄積ステップ数（実質的なバッチサイズ = 2 x 8 = 16）
        # 学習スケジュール
        warmup_steps = 5,               # ウォームアップステップ数
        num_train_epochs = 2,           # エポック数
        #max_steps = 60, # デモ用 (本番ではコメントアウトし num_train_epochs = 2 等に設定)

        # 最適化設定
        learning_rate = 2e-4,           # 学習率
        logging_steps = 10,             # ログ出力頻度
        optim = "adamw_8bit",           # 8ビットAdamWオプティマイザを使用
        lr_scheduler_type = "cosine",   # 学習率スケジューラー(指定しないとlinear)

        # その他
        fp16 = use_fp16,
        bf16 = use_bf16,
        output_dir = BASE + "outputs",
        report_to=[],
        seed = 3407,                  # 乱数シード（指定しないとランダム）
    ),
)

# 学習
trainer.train()

# LoRAアダプターのみを保存
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"LoRAアダプターを {save_dir} に保存しました")

# メモリクリア（Google Colab等での連続実行時に備えて）
del model
del trainer
del dataset
del tokenizer
import gc
gc.collect()
torch.cuda.empty_cache()
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")