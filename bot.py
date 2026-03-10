import json
import os
import re
import textwrap
from datetime import datetime
from urllib.parse import quote
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
    "mistral",
    "mixtral",
    "phi",
    "vllm",
    "rag",
    "ai agent",
    "agentic",
    "inference",
    "gpu",
    "cuda",
    "tensorrt",
    "open source ai",
    "open-source ai",
]

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MAX_INITIAL_CANDIDATES = 20
MAX_SEARCH_QUERIES = 3
MAX_SEARCH_RESULTS_PER_QUERY = 5
MAX_FINAL_CANDIDATES = 30
MAX_OUTPUT_ITEMS = 5
HISTORY_FILE = "sent_articles.json"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AI-News-Bot/1.0)"
}


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


def extract_json_object(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("JSONオブジェクトを抽出できませんでした。")


def clean_summary_text(summary: str) -> str:
    summary = re.sub(r"<[^>]+>", " ", summary or "")
    summary = re.sub(r"\s+", " ", summary).strip()
    return summary


def resolve_final_url(url: str) -> str:
    """
    Google News のような中継URLを、できるだけ最終URLに解決する。
    失敗したら元URLを返す。
    """
    if not url:
        return url

    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=15,
            allow_redirects=True,
        )
        final_url = str(response.url).strip()
        if final_url:
            return final_url
    except requests.RequestException:
        pass

    return url


def suppress_discord_embeds(text: str) -> str:
    """
    Discordの埋め込みカードを抑えるため、
    URLを <...> で囲む。
    """
    def repl(match):
        url = match.group(0)
        if url.startswith("<") and url.endswith(">"):
            return url
        return f"<{url}>"

    return re.sub(r"https?://[^\s>]+", repl, text)


def fetch_feed_items(feed_urls: list[str], sent_history: set[str]) -> list[dict]:
    items = []

    for url in feed_urls:
        feed = feedparser.parse(url)
        source_name = getattr(feed.feed, "title", url)

        for entry in feed.entries:
            title = getattr(entry, "title", "")
            summary = clean_summary_text(getattr(entry, "summary", ""))
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

    return items


def dedupe_items(items: list[dict], limit: int) -> list[dict]:
    seen_title = set()
    seen_link = set()
    unique_items = []

    for item in sorted(items, key=lambda x: x["score"], reverse=True):
        title_key = normalize_text(item["title"])
        link_key = item["link"].strip()

        if title_key in seen_title or link_key in seen_link:
            continue

        seen_title.add(title_key)
        seen_link.add(link_key)
        unique_items.append(item)

    return unique_items[:limit]


def fetch_initial_candidates(sent_history: set[str]) -> list[dict]:
    items = fetch_feed_items(RSS_FEEDS, sent_history)
    return dedupe_items(items, MAX_INITIAL_CANDIDATES)


def build_search_query_prompt(items: list[dict]) -> str:
    bullets = []
    for i, item in enumerate(items, start=1):
        bullets.append(
            f"[{i}] {item['title']} | source={item['source']} | link={item['link']}"
        )

    joined = "\n".join(bullets)

    return textwrap.dedent(
        f"""
        あなたはAIニュースの調査担当です。
        以下の候補記事を見て、追加調査すべき検索クエリを最大{MAX_SEARCH_QUERIES}個だけ作成してください。

        条件:
        - 英語の短い検索クエリ
        - AI/ローカルLLM/GPU/AIエージェント関連を優先
        - 曖昧すぎる語は避ける
        - 重複クエリは避ける
        - JSONのみで返す
        - 形式:
        {{
          "queries": ["query1", "query2"]
        }}

        候補記事:
        {joined}
        """
    ).strip()


def propose_search_queries_with_openai(client: OpenAI, items: list[dict]) -> list[str]:
    if not items:
        return []

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=build_search_query_prompt(items),
    )

    text = response.output_text.strip() if getattr(response, "output_text", None) else ""
    data = extract_json_object(text)
    queries = data.get("queries", [])

    cleaned = []
    for q in queries:
        q = str(q).strip()
        if q and q not in cleaned:
            cleaned.append(q)

    return cleaned[:MAX_SEARCH_QUERIES]


def build_google_news_rss_url(query: str) -> str:
    encoded = quote(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


def fetch_search_results(queries: list[str], sent_history: set[str]) -> list[dict]:
    items = []

    for query in queries:
        rss_url = build_google_news_rss_url(query)
        feed = feedparser.parse(rss_url)
        source_name = f"Google News Search: {query}"

        count = 0
        for entry in feed.entries:
            if count >= MAX_SEARCH_RESULTS_PER_QUERY:
                break

            title = getattr(entry, "title", "")
            summary = clean_summary_text(getattr(entry, "summary", ""))
            raw_link = getattr(entry, "link", "")

            if not raw_link:
                continue

            final_link = resolve_final_url(raw_link)

            if final_link in sent_history:
                continue

            scored = score_entry(title, summary)
            if scored <= 0:
                scored = 1

            items.append(
                {
                    "title": title,
                    "summary": summary,
                    "link": final_link,
                    "score": scored + 1,
                    "source": source_name,
                }
            )
            count += 1

    return items


def build_selection_prompt(items: list[dict]) -> str:
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
        あなたはAIニュース編集長です。
        候補記事から、今日送るべき重要ニュースを最大{MAX_OUTPUT_ITEMS}件選んでください。

        条件:
        - AI、ローカルLLM、GPU推論、AIエージェント、OSS AIを優先
        - 同じ話題の重複は避ける
        - JSONのみで返す
        - index は上の候補番号を使う
        - 形式:
        {{
          "selected_indexes": [1, 3, 5]
        }}

        候補記事:
        {joined}
        """
    ).strip()


def choose_top_items_with_openai(client: OpenAI, items: list[dict]) -> list[dict]:
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=build_selection_prompt(items),
    )

    text = response.output_text.strip() if getattr(response, "output_text", None) else ""
    data = extract_json_object(text)
    indexes = data.get("selected_indexes", [])

    selected = []
    for idx in indexes:
        try:
            i = int(idx) - 1
            if 0 <= i < len(items):
                selected.append(items[i])
        except (ValueError, TypeError):
            continue

    return dedupe_items(selected, MAX_OUTPUT_ITEMS)


def build_summary_prompt(selected_items: list[dict], queries: list[str]) -> str:
    bullets = []
    for i, item in enumerate(selected_items, start=1):
        bullets.append(
            textwrap.dedent(
                f"""
                [{i}]
                title: {item['title']}
                source: {item['source']}
                link: {item['link']}
                summary: {item['summary']}
                """
            ).strip()
        )

    joined = "\n\n".join(bullets)
    query_text = ", ".join(queries) if queries else "なし"

    return textwrap.dedent(
        f"""
        あなたはAI/ローカルLLM/GPUニュース専門の編集者です。
        以下の選定済み記事を、Discord向けに日本語で見やすく要約してください。

        ルール:
        - 日本語で出力
        - 誇張しない
        - 1記事あたり「見出し + 箇条書き2つ + URL」にする
        - 箇条書きは短めにする
        - 「何が起きたか」は1行
        - 「なぜ重要か」は1行
        - 無駄な前置きは不要
        - URLはそのまま1行で出す

        出力形式:
        1. 見出し
        ・何が起きたか
        ・なぜ重要か
        ・URL

        最後に必ず以下を入れる:
        ---
        今日のひとこと:
        〇〇

        検索で深掘りしたクエリ:
        {query_text}

        選定済み記事:
        {joined}
        """
    ).strip()


def summarize_selected_items_with_openai(
    client: OpenAI, selected_items: list[dict], queries: list[str]
) -> str:
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=build_summary_prompt(selected_items, queries),
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

    safe_message = suppress_discord_embeds(message)
    chunks = split_message_for_discord(safe_message)

    for chunk in chunks:
        res = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": chunk},
            timeout=30,
        )
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

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")

    client = OpenAI(api_key=OPENAI_API_KEY)
    sent_history = load_history()

    initial_items = fetch_initial_candidates(sent_history)
    print(f"初期候補記事数: {len(initial_items)}")

    if not initial_items:
        print("新しい初期候補記事が見つかりませんでした。")
        return

    search_queries = propose_search_queries_with_openai(client, initial_items)
    print(f"追加検索クエリ: {search_queries}")

    search_items = fetch_search_results(search_queries, sent_history)
    print(f"追加検索記事数: {len(search_items)}")

    all_candidates = dedupe_items(initial_items + search_items, MAX_FINAL_CANDIDATES)
    print(f"最終候補記事数: {len(all_candidates)}")

    selected_items = choose_top_items_with_openai(client, all_candidates)
    print(f"選定記事数: {len(selected_items)}")

    if not selected_items:
        print("送信対象の記事が選ばれませんでした。")
        return

    digest = summarize_selected_items_with_openai(client, selected_items, search_queries)
    header = f"📡 AIニュース速報 ({now_jst_str()})\n"
    final_message = header + "\n" + digest

    print(final_message)
    post_to_discord(final_message)

    update_history(selected_items)
    print("履歴を更新しました。")
    print("完了")


if __name__ == "__main__":
    main()
