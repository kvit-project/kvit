"""OpenAI-compatible proxy in front of the patched vLLM.

Tool outputs that are slices of files tracked in the repo become PIC spans, so
vLLM caches their KV the first time the agent reads them and reuses it at any
position later. Nothing is prefilled ahead of time.
"""

import argparse
import json
import os
import re
import subprocess

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from transformers import AutoTokenizer

# "   12\t" (cat -n, Claude Code Read) or "12→" line-number prefixes
LINE_NUMBER = re.compile(r"^ *\d+(?:\t|→)", re.MULTILINE)
MAX_FILE_BYTES = 1 << 20
# chat fields /v1/completions does not take, or takes with another meaning
CHAT_ONLY = {
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "chat_template_kwargs",
    "stream",
    "stream_options",
    "logprobs",
    "top_logprobs",
}


class RepoIndex:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self._cache: dict[str, tuple[int, str]] = {}

    def _files(self):
        out = subprocess.run(
            ["git", "-C", self.root, "ls-files", "-z"],
            capture_output=True,
            check=True,
        ).stdout.decode()
        return [os.path.join(self.root, p) for p in out.split("\0") if p]

    def _text(self, path: str) -> str:
        try:
            st = os.stat(path)
        except OSError:
            return ""
        hit = self._cache.get(path)
        if hit and hit[0] == st.st_mtime_ns:
            return hit[1]
        text = ""
        if st.st_size <= MAX_FILE_BYTES:
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError):
                pass
        self._cache[path] = (st.st_mtime_ns, text)
        return text

    def contains(self, output: str) -> bool:
        """True if the output, minus line-number prefixes, is a slice of a file."""
        body = LINE_NUMBER.sub("", output).strip()
        # ponytail: scans every tracked file per tool output, index lines if slow
        return bool(body) and any(body in self._text(p) for p in self._files())


def split_prompt(text: str, spans: list[str]) -> list[tuple[str, bool]]:
    """Cut the rendered prompt into (text, is_span) pieces, spans in order."""
    pieces, pos = [], 0
    for span in spans:
        i = text.find(span, pos)
        if i < 0:
            continue
        pieces += [(text[pos:i], False), (span, True)]
        pos = i + len(span)
    pieces.append((text[pos:], False))
    return pieces


def layout(
    pieces: list[tuple[list[int], bool]], block_size: int, pad_id: int
) -> tuple[list[int], list[int], list[int]]:
    """Concatenate token pieces, padding so each span and the text after it
    start on a block boundary. Returns ids, span_starts, cross_span_starts."""
    ids: list[int] = []
    starts: list[int] = []
    crosses: list[int] = []
    after_span = False
    for toks, is_span in pieces:
        if not toks:
            continue
        # a span shorter than a block can never be reused
        is_span = is_span and len(toks) >= block_size
        if is_span or after_span:
            ids += [pad_id] * (-len(ids) % block_size)
            (starts if is_span else crosses).append(len(ids))
        ids += toks
        after_span = is_span
    return ids, starts, crosses


def text_content(message: dict) -> dict:
    content = message.get("content")
    if isinstance(content, list) and all(p.get("type") == "text" for p in content):
        message = {**message, "content": "\n".join(p["text"] for p in content)}
    return message


def create_app(repo: RepoIndex, vllm_url: str, block_size: int) -> FastAPI:
    client = httpx.Client(base_url=vllm_url, timeout=None)
    aclient = httpx.AsyncClient(base_url=vllm_url, timeout=None)
    model_path = client.get("/v1/models").raise_for_status().json()["data"][0]["root"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pad_id = tokenizer("\n", add_special_tokens=False).input_ids[-1]
    app = FastAPI()

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        messages = [text_content(m) for m in body["messages"]]
        tools = body.get("tools")
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **(body.get("chat_template_kwargs") or {}),
        )
        spans = [
            m["content"]
            for m in messages
            if m.get("role") == "tool"
            and isinstance(m.get("content"), str)
            and repo.contains(m["content"])
        ]
        pieces = [
            (tokenizer(t, add_special_tokens=False).input_ids, s)
            for t, s in split_prompt(text, spans)
        ]
        ids, starts, crosses = layout(pieces, block_size, pad_id)
        print(f"kvit: {len(starts)} spans, {len(ids)} prompt tokens", flush=True)

        req = {k: v for k, v in body.items() if k not in CHAT_ONLY}
        if "max_completion_tokens" in req:
            req["max_tokens"] = req.pop("max_completion_tokens")
        req["prompt"] = ids
        if starts:
            req["vllm_xargs"] = {
                **(req.get("vllm_xargs") or {}),
                "span_starts": starts,
                "cross_span_starts": crosses,
            }
        r = client.post("/v1/completions", json=req)
        if r.status_code != 200:
            return Response(r.content, r.status_code, media_type="application/json")
        out = r.json()
        # ponytail: only choice 0, n > 1 is dropped
        choice = out["choices"][0]
        parsed = (
            client.post(
                "/v1/chat/completions/parse",
                json={
                    "text": choice["text"],
                    "model": body.get("model"),
                    "tools": tools,
                },
            )
            .raise_for_status()
            .json()
        )

        message = {
            "role": "assistant",
            "content": parsed["content"],
            "reasoning_content": parsed["reasoning_content"],
            "tool_calls": [
                {**tc, "index": i} for i, tc in enumerate(parsed["tool_calls"])
            ]
            or None,
        }
        finish = "tool_calls" if parsed["tools_called"] else choice["finish_reason"]
        resp = {
            "id": out["id"],
            "object": "chat.completion",
            "created": out["created"],
            "model": out["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": out.get("usage"),
        }
        if not body.get("stream"):
            return resp
        # ponytail: the whole answer arrives as one chunk, no token streaming
        chunk = {
            **resp,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": message, "finish_reason": finish}],
        }
        return StreamingResponse(
            iter([f"data: {json.dumps(chunk)}\n\n", "data: [DONE]\n\n"]),
            media_type="text/event-stream",
        )

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def passthrough(path: str, request: Request):
        r = await aclient.request(
            request.method,
            "/" + path,
            content=await request.body(),
            headers={"content-type": request.headers.get("content-type", "")},
        )
        return Response(
            r.content, r.status_code, media_type=r.headers.get("content-type")
        )

    return app


def main():
    parser = argparse.ArgumentParser(prog="kvit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="OpenAI-compatible proxy for an agent")
    serve.add_argument("--repo", required=True, help="local git repository")
    serve.add_argument("--vllm-url", default="http://localhost:8000")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)
    serve.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    app = create_app(RepoIndex(args.repo), args.vllm_url, args.block_size)
    uvicorn.run(app, host=args.host, port=args.port)
