import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import os

max_seq_length = 2048 # シーケンス長を設定
model_name = os.environ.get('MODEL', 'unsloth/Llama-3.2-3B-Instruct')
BASE = vars().get('BASE', './')
save_dir = BASE + "finetuned_model"
merged_save_dir = BASE + "merged_model_full"

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    compute_dtype = torch.bfloat16
    use_bf16 = True
    use_fp16 = False
else:
    compute_dtype = torch.float16
    use_bf16 = False
    use_fp16 = True
print(f"compute_dtype: {compute_dtype} use_bf16: {use_bf16} use_fp16: {use_fp16}")

print("=== 完全なマージ処理を開始 ===")

# 16ビット精度でベースモデルを再読み込み
print("ステップ1: ベースモデルを16ビットで読み込んでいます...")
model_16bit, tokenizer_16bit = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = compute_dtype,
    load_in_4bit = False,  # 重要: Falseに設定
    #device_map="cpu",      # CPUにロード
)
print("ベースモデル読み込み完了")

# LoRAアダプターを読み込み
print("ステップ2: LoRAアダプターを読み込んでいます...")
model_with_lora = PeftModel.from_pretrained(model_16bit, save_dir)
print("LoRAアダプター読み込み完了")

# マージ実行
print("ステップ3: マージを実行しています...")
merged_model = model_with_lora.merge_and_unload()
print("マージ完了")

# 保存
print("ステップ4: 保存しています（数分かかります）...")
merged_model.save_pretrained(merged_save_dir)
tokenizer_16bit.save_pretrained(merged_save_dir)
print(f"保存完了: {merged_save_dir}")

# サイズ確認
total_size = sum(os.path.getsize(os.path.join(merged_save_dir, f)) 
                 for f in os.listdir(merged_save_dir) if os.path.isfile(os.path.join(merged_save_dir, f)))
print(f"保存されたモデルのサイズ: {total_size / (1024 * 1024 * 1024):.2f} GB")

# メモリクリア
del model_16bit
del model_with_lora
del merged_model
del tokenizer_16bit
import gc
gc.collect()
torch.cuda.empty_cache()
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")