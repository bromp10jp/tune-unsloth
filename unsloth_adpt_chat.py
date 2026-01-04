import os
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

# --- 設定（必要に応じて変更してください） ---
BASE = vars().get('BASE', './')
model_name = os.environ.get('MODEL', 'unsloth/Llama-3.2-3B-Instruct')  # 元モデル
adapter_dir = BASE + "finetuned_model"  # LoRAアダプタの保存ディレクトリ
max_seq_length = 2048
max_new_tokens = 512
temperature = 0.0
top_p = 0.95

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# dtype の選定（推論用）
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    dtype = torch.bfloat16
else:
    dtype = torch.float16

print(f"dtype for loading: {dtype}")

# ベースモデルの読み込み（都度ダウンロード）
print(f"ベースモデル '{model_name}' を読み込み中...")
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True,  # 学習時と同じ4ビット量子化を使用
)
print("ベースモデル読み込み完了")

# LoRAアダプタを読み込んでモデルに適用
print(f"LoRAアダプタ '{adapter_dir}' を読み込み中...")
model = PeftModel.from_pretrained(base_model, adapter_dir)
print("LoRAアダプタ読み込み完了")

# 推論モードに設定
model.eval()
print("モデル準備完了")

# チャット履歴を保持するためのリスト
system_prompt = "あなたは優秀なガイドです。"
chat_history = [{"role": "system", "content": system_prompt}]

print("終了するには 'exit' または 'quit' を入力してください。")
print("会話をリセットするには 'reset' を入力してください。")
print("ユーザ入力を待っています...")

while True:
    user_input = input("\nUser: ").strip()
    if not user_input:
        continue

    low = user_input.lower()
    if low in ("exit", "quit"):
        print("終了します。")
        break

    if low == "reset":
        # system_prompt を保持して会話履歴を新規にする
        chat_history = [{"role": "system", "content": system_prompt}]
        print("会話をリセットしました。新しい会話を開始してください。")
        continue

    # 履歴にユーザ発話を追加
    chat_history.append({"role": "user", "content": user_input})

    # apply_chat_template を使ってモデル入力テキストを作成
    input_text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

    # トークナイズしてテンソル化
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # 生成
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 生成トークン列から、入力トークン長をスキップして応答部分のみデコード
    gen_tokens = gen_ids[0]
    input_len = input_ids.shape[1]
    if gen_tokens.shape[0] > input_len:
        response_tokens = gen_tokens[input_len:]
    else:
        response_tokens = torch.tensor([], dtype=torch.long)

    assistant_response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    if assistant_response == "":
        assistant_response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # 履歴にアシスタント発話を追加
    chat_history.append({"role": "assistant", "content": assistant_response})

    # 表示
    print("\nAssistant:", assistant_response)

# メモリクリア
del model
del base_model
del tokenizer
import gc
gc.collect()
torch.cuda.empty_cache()
print("終了しました。")
