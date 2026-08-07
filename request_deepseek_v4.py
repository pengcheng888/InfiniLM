#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send one or more OpenAI-compatible chat requests to InfiniLM."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--url", default=None, help="Full base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek_v4")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="User prompt. Pass more than once to send sequential requests.",
    )
    parser.add_argument("--system", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1000.0)
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false")
    return parser.parse_args()


def build_payload(args, prompt):
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})
    return {
        "model": args.model,
        "messages": messages,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }


def post_json(url, payload, timeout):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def run_stream(url, payload, timeout):
    answer = []
    with post_json(url, payload, timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            choice = event.get("choices", [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or choice.get("text") or ""
            if content:
                print(content, end="", flush=True)
                answer.append(content)
            if choice.get("finish_reason") is not None:
                break
    print()
    return "".join(answer)


def run_non_stream(url, payload, timeout):
    with post_json(url, payload, timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    print(content)
    return content


def main():
    args = parse_args()
    prompts = args.prompt or ["Say hello in one sentence."]
    base_url = args.url or f"http://{args.host}:{args.port}"
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"

    all_ok = True
    for i in range(args.repeat):
        for idx, prompt in enumerate(prompts, start=1):
            print(f"=== Request {i * len(prompts) + idx} ===")
            print(f"User: {prompt}")
            print("Assistant: ", end="", flush=True)
            payload = build_payload(args, prompt)
            try:
                if args.stream:
                    answer = run_stream(endpoint, payload, args.timeout)
                else:
                    answer = run_non_stream(endpoint, payload, args.timeout)
                if not answer.strip():
                    all_ok = False
                    print("Empty response", file=sys.stderr)
            except urllib.error.HTTPError as exc:
                all_ok = False
                body = exc.read().decode("utf-8", errors="replace")
                print(f"HTTP {exc.code}: {body}", file=sys.stderr)
            except Exception as exc:
                all_ok = False
                print(f"Request failed: {exc}", file=sys.stderr)
            if args.interval > 0:
                time.sleep(args.interval)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
