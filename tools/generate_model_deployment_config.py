#!/usr/bin/env python3
"""
generate_model_deployment_config.py - Interactive helper for v7.0 model topology.

Produces a `model_deployment.json` that Phoenix can load with
--model-deployment-config / AI_MODEL_DEPLOYMENT_CONFIG.

Example:
    python tools/generate_model_deployment_config.py --llm remote \
        --llm-url http://192.168.1.10:11434 --llm-method ollama \
        --llm-model llama3.1:8b \
        --vision remote --vision-url http://192.168.1.11:5000/infer \
        --speech remote --speech-url http://192.168.1.12:5001/infer \
        -o config/model_deployment.json
"""

import argparse
import json
import sys


def build_record(placement: str, url: str = "", method: str = "",
                 model: str = "", timeout: int = 30000) -> dict:
    record = {"placement": placement}
    if placement in ("remote", "auto"):
        record["remote"] = {
            "url": url,
            "method": method,
            "modelName": model,
            "authToken": "",
            "timeoutMs": timeout,
            "headers": {},
        }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Phoenix v7.0 model deployment config")
    parser.add_argument("--llm", default="local", choices=["local", "remote", "auto"],
                        help="LLM placement")
    parser.add_argument("--llm-url", default="", help="LLM remote URL")
    parser.add_argument("--llm-method", default="ollama",
                        choices=["ollama", "llamacpp", "bitnet"],
                        help="LLM remote backend method")
    parser.add_argument("--llm-model", default="", help="LLM remote model name")
    parser.add_argument("--llm-timeout", type=int, default=120000,
                        help="LLM remote timeout in ms")

    parser.add_argument("--vision", default="local", choices=["local", "remote", "auto"],
                        help="Vision placement")
    parser.add_argument("--vision-url", default="", help="Vision remote URL")
    parser.add_argument("--vision-method", default="http-json",
                        help="Vision remote method")
    parser.add_argument("--vision-timeout", type=int, default=30000,
                        help="Vision remote timeout in ms")

    parser.add_argument("--speech", default="local", choices=["local", "remote", "auto"],
                        help="Speech placement")
    parser.add_argument("--speech-url", default="", help="Speech remote URL")
    parser.add_argument("--speech-method", default="http-json",
                        help="Speech remote method")
    parser.add_argument("--speech-timeout", type=int, default=30000,
                        help="Speech remote timeout in ms")

    parser.add_argument("-o", "--output", default="config/model_deployment.json",
                        help="Output file path")
    args = parser.parse_args()

    config = {
        "llm": build_record(args.llm, args.llm_url, args.llm_method,
                            args.llm_model, args.llm_timeout),
        "vision": build_record(args.vision, args.vision_url, args.vision_method,
                               "", args.vision_timeout),
        "speech": build_record(args.speech, args.speech_url, args.speech_method,
                               "", args.speech_timeout),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
