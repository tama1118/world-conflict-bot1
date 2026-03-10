import json
import os
import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo

import feedparser
import requests
from openai import OpenAI

RSS_FEEDS = [
    "https://feeds.feedburner.com/venturebeat/SZYF",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://www.tomshardware.com/feeds/all",
]

KEYWORDS = [
    "local llm",
    "ollama",
    "lm studio",
    "qwen",
    "deepseek",
    "llama",
    "gemma",
    "vllm",
    "rag",
    "ai agent",
    "agentic",
    "inference",
    "gpu",
    "cuda",
    "open source ai",
    "open-source ai",
]

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MAX_CANDIDATES = 25
MAX_OUTPUT_ITEMS = 5
HISTORY_FILE = "sent_articles.json"


def now_jst_str() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S JST")


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def load_history() -> set[str]:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return set(str(x) for x in data)
            return set()
    except FileNotFoundError:
        return set()
    except json.JSONDecodeError:
        return set()


def save_history(history: set[str]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(history), f, ensure_ascii=False, indent=2)


def score_entry(title: str, summary: str) -> int:
    text = normalize_text(f"{title} {summary}")
    score = 0
    for kw in KEYWORDS:
        if kw in text:
            score += 1
    return score


def fetch_rss_items() -> list[dict]:
    sent_history = load_history()
    items = []

    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        source_name = getattr(feed.feed, "title", url)

        for entry in feed.entries:
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            link = getattr(entry, "link", "")

            if not link or link in sent_history:
                continue

            scored = score_entry(title, summary)
            if scored <= 0:
                continue

            items.append(
                {
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "score": scored,
                    "source": source_name,
                }
            )

    seen_titles = set()
    unique_items = []
    for item in sorted(items, key=lambda x: x["score"], reverse=True):
        key = normalize_text(item["title"])
        if key in seen_titles:
            continue
        seen_titles.add(key)
        unique_items.append(item)

    return unique_items[:MAX_CANDIDATES]


def build_prompt(items: list[dict]) -> str:
    bullets = []
    for i, item in enumerate(items, start=1):
        bullets.append(
            textwrap.dedent(
                f"""
                [{i}]
                title: {item['title']}
                source: {item['source']}
                link: {item['link']}
                summary: {item['summary']}
                score: {item['score']}
                """
            ).strip()
        )

    joined = "\n\n".join(bullets)

    return textwrap.dedent(
        f"""
        あなたはAI/ローカルLLM/GPUニュース専門の編集者です。
        候補記事の中から重要度の高いものを最大{MAX_OUTPUT_ITEMS}件選び、
        Discord向けに日本語で簡潔にまとめてください。

        ルール:
        - 日本語で出力
        - 誇張しない
        - ローカルLLM、GPU推論、AIエージェント、OSS AI を優先
        - 同じ話題はまとめる
        - 出力形式は必ず以下

        1. 見出し
        ・何が起きたか
        ・なぜ重要か
        ・URL

        2. 見出し
        ・何が起きたか
        ・なぜ重要か
        ・URL

        最後に必ず以下を入れる:
        ---
        今日のひとこと:
        〇〇

        候補記事:
        {joined}
        """
    ).strip()


def summarize_with_openai(items: list[dict]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_prompt(items)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    texts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                texts.append(text)

    if texts:
        return "\n".join(texts).strip()

    raise RuntimeError("OpenAI response をテキスト化できませんでした。")


def split_message_for_discord(message: str, max_len: int = 1800) -> list[str]:
    chunks = []
    current = ""

    for line in message.splitlines(True):
        if len(current) + len(line) > max_len:
            if current:
                chunks.append(current)
            current = line
        else:
            current += line

    if current:
        chunks.append(current)

    return chunks


def post_to_discord(message: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL 未設定のため、Discord送信をスキップします。")
        return

    chunks = split_message_for_discord(message)
    for chunk in chunks:
        res = requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk}, timeout=30)
        res.raise_for_status()


def update_history(sent_items: list[dict]) -> None:
    history = load_history()
    for item in sent_items:
        link = item.get("link", "")
        if link:
            history.add(link)
    save_history(history)


def main() -> None:
    print(f"[{now_jst_str()}] ニュース収集開始")
    print(f"OPENAI_MODEL={OPENAI_MODEL}")
    print(f"OPENAI_API_KEY_SET={bool(OPENAI_API_KEY)}")
    print(f"DISCORD_WEBHOOK_URL_SET={bool(DISCORD_WEBHOOK_URL)}")

    items = fetch_rss_items()
    print(f"候補記事数: {len(items)}")

    if not items:
        print("新しい候補記事が見つかりませんでした。")
        return

    digest = summarize_with_openai(items)
    header = f"📡 AIニュース速報 ({now_jst_str()})\n"
    final_message = header + "\n" + digest

    print(final_message)
    post_to_discord(final_message)

    sent_items = items[:MAX_OUTPUT_ITEMS]
    update_history(sent_items)

    print("履歴を更新しました。")
    print("完了")


if __name__ == "__main__":
    main()
