import pandas as pd
from openai import OpenAI
from loguru import logger
import sys, os, datetime, re
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

# Excel 列名（保持中文不变）
LANGS = [
    "英语", "法语", "德语", "意大利语", "西班牙语", "俄语", "葡萄牙语", "捷克语",
    "日语", "斯洛伐克语", "波兰语", "匈牙利语", "荷兰语", "乌克兰语", "阿拉伯语"
]

# 给模型用的标准目标语言名（避免 “Translate to 英语” 这类混搭导致跑偏）
LANG_EN = {
    "英语": "English",
    "法语": "French",
    "德语": "German",
    "意大利语": "Italian",
    "西班牙语": "Spanish",
    "俄语": "Russian",
    "葡萄牙语": "Portuguese",
    "捷克语": "Czech",
    "日语": "Japanese",
    "斯洛伐克语": "Slovak",
    "波兰语": "Polish",
    "匈牙利语": "Hungarian",
    "荷兰语": "Dutch",
    "乌克兰语": "Ukrainian",
    "阿拉伯语": "Arabic",
}

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 日志仅记录错误到文件（每次运行覆盖旧日志，避免历史残留误判）
logger.remove()
logger.add("error_log.log", level="ERROR", mode="w")

# --- 轻量语言校验（用于“翻成中文”等跑偏） ---
_RE_HAN = re.compile(r"[\u4e00-\u9fff]")     # 汉字
_RE_ARABIC = re.compile(r"[\u0600-\u06FF]") # 阿拉伯字符
_RE_CYR = re.compile(r"[\u0400-\u04FF]")    # 西里尔（俄/乌）
_RE_KANA = re.compile(r"[\u3040-\u30FF]")   # 日语假名（平/片）

def _lang_ok(lang_cn: str, out: str) -> bool:
    s = (out or "").strip()
    if not s:
        return True

    # 目标不是日语：出现汉字 => 判为跑偏，触发重试
    if lang_cn != "日语" and _RE_HAN.search(s):
        return False

    # 阿拉伯语：必须包含阿拉伯字符（否则大概率跑偏）
    if lang_cn == "阿拉伯语" and not _RE_ARABIC.search(s):
        return False

    # 俄语 / 乌克兰语：建议至少包含西里尔（技术文本可能夹英文，但完全没有通常不对）
    if lang_cn in ("俄语", "乌克兰语") and not _RE_CYR.search(s):
        return False

    # 日语：不强杀（因为日语可含汉字/也可能纯英文术语）
    if lang_cn == "日语":
        return True

    return True

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_api(text: str, lang_cn: str):
    lang_en = LANG_EN.get(lang_cn, lang_cn)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional technical translator. "
                f"Translate the user's text into {lang_en}. "
                f"Return ONLY the translation, written in {lang_en}."
            )
        },
        {"role": "user", "content": text}
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        timeout=30
    )
    out_text = response.choices[0].message.content.strip()

    # 校验失败 => 抛异常触发 tenacity 自动重试（不增加额外 API 调用）
    if not _lang_ok(lang_cn, out_text):
        raise ValueError(f"LANG_MISMATCH: expected={lang_en}({lang_cn})")

    return {
        "text": out_text,
        "in": getattr(response.usage, "prompt_tokens", 0) or 0,
        "out": getattr(response.usage, "completion_tokens", 0) or 0
    }

def do_job(row_idx: int, lang_cn: str, text):
    if pd.isna(text) or str(text).strip() == "":
        return row_idx, lang_cn, "", 0, 0

    # ✅ 最小修复：英语列不走 API，直接回填原文（省钱 + 100% 成功率）
    if lang_cn == "英语":
        return row_idx, lang_cn, str(text), 0, 0

    try:
        res = call_api(str(text), lang_cn)
        return row_idx, lang_cn, res["text"], res["in"], res["out"]
    except Exception as e:
        logger.error(f"Error at Row {row_idx} [{lang_cn}]: {e}")

        # ✅ 兜底：如果英语列出现 ERROR（理论上不会走到这里），强制回填原文
        if lang_cn == "英语":
            return row_idx, lang_cn, str(text), 0, 0

        return row_idx, lang_cn, "ERROR", 0, 0

def main():
    print(f"\n{'='*50}\n🚀 DeepSeek 工业翻译官 (稳定列顺序 + 语言校验 + 英语直拷贝)\n{'='*50}")

    input_file = sys.argv[1] if len(sys.argv) > 1 else "source.xlsx"
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到文件 '{input_file}'")
        return

    df = pd.read_excel(input_file)

    if "Original" not in df.columns:
        print("❌ 错误: Excel 必须包含列名 'Original'")
        return

    # ✅ 先按固定顺序创建语言列，避免 as_completed 导致列顺序漂移
    for lang in LANGS:
        if lang not in df.columns:
            df[lang] = ""

    total_tasks = len(df) * len(LANGS)

    # 统计数据
    stats = {"in": 0, "out": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx in range(len(df)):
            source = df.at[idx, "Original"]
            for lang in LANGS:
                futures.append(executor.submit(do_job, idx, lang, source))

        with tqdm(total=total_tasks, desc="任务进度", unit="格", colour="#00ff00") as pbar:
            for f in as_completed(futures):
                r_idx, lang, res, in_t, out_t = f.result()
                df.at[r_idx, lang] = res
                stats["in"] += in_t
                stats["out"] += out_t
                pbar.update(1)

    # ✅ 保存前强制重排列顺序：Original + LANGS，其它列保留在最后
    head = ["Original"] + LANGS
    tail = [c for c in df.columns if c not in head]
    df = df[head + tail]

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

