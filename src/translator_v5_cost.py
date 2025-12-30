import pandas as pd
from openai import OpenAI
from loguru import logger
import sys, os, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from dotenv import load_dotenv

# ==========================================
# ⚙️ 配置与安全 (Env & Config)
# ==========================================
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 2025 DeepSeek-V3 官方定价 (人民币)
PRICE_IN_M = 2.0   # 输入: 2元/百万 tokens
PRICE_OUT_M = 8.0  # 输出: 8元/百万 tokens

MAX_WORKERS = 45   # 性能模式，Tier 1 用户若报错可降至 15
LANGS = ["英语", "法语", "德语", "意大利语", "西班牙语", "俄语", "葡萄牙语", "捷克语", "日语", "斯洛伐克语", "波兰语", "匈牙利语", "荷兰语", "乌克兰语", "阿拉伯语"]

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 日志仅记录错误到文件
logger.remove()
logger.add("error_log.log", level="ERROR")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_api(text, lang):
    messages = [
        {"role": "system", "content": f"You are a professional technical translator. Translate to {lang}. Return ONLY the result."},
        {"role": "user", "content": text}
    ]
    response = client.chat.completions.create(model="deepseek-chat", messages=messages, timeout=30)
    return {
        "text": response.choices[0].message.content.strip(),
        "in": response.usage.prompt_tokens,
        "out": response.usage.completion_tokens
    }

def do_job(row_idx, lang, text):
    if pd.isna(text) or str(text).strip() == "":
        return row_idx, lang, "", 0, 0
    try:
        res = call_api(text, lang)
        return row_idx, lang, res["text"], res["in"], res["out"]
    except Exception as e:
        logger.error(f"Error at Row {row_idx} [{lang}]: {e}")
        return row_idx, lang, "ERROR", 0, 0

def main():
    print(f"\n{'='*50}\n🚀 DeepSeek-V3 工业翻译官 (Pro版)\n{'='*50}")
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "source.xlsx"
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到文件 '{input_file}'"); return

    df = pd.read_excel(input_file)
    total_tasks = len(df) * len(LANGS)
    
    # 统计数据
    stats = {"in": 0, "out": 0}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx in range(len(df)):
            source = df.at[idx, 'Original']
            for lang in LANGS:
                futures.append(executor.submit(do_job, idx, lang, source))
        
        with tqdm(total=total_tasks, desc="任务进度", unit="格", colour="#00ff00") as pbar:
            for f in as_completed(futures):
                r_idx, lang, res, in_t, out_t = f.result()
                df.at[r_idx, lang] = res
                stats["in"] += in_t
                stats["out"] += out_t
                pbar.update(1)

    # 费用结算
    cost_in = (stats["in"] / 1_000_000) * PRICE_IN_M
    cost_out = (stats["out"] / 1_000_000) * PRICE_OUT_M
    
    print(f"\n{'💰 账单结算':-^40}")
    print(f"输入消耗: {stats['in']:>8} tokens (￥{cost_in:.4f})")
    print(f"输出消耗: {stats['out']:>8} tokens (￥{cost_out:.4f})")
    print(f"总计成本: ￥{cost_in + cost_out:.4f}")
    print("-" * 40)

    out_name = f"Translated_{datetime.datetime.now().strftime('%m%d_%H%M')}.xlsx"
    df.to_excel(out_name, index=False)
    print(f"✨ 处理完成！结果已存至: {out_name}\n")

if __name__ == "__main__":
    main()