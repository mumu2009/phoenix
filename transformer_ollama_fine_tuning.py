#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict, List, Sequence


COMMON_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "W_pack",
    "wq",
    "wk",
    "wv",
    "wo",
    "c_attn",
    "c_proj",
    "query_key_value",
    "dense",
    "fc1",
    "fc2",
)


@dataclass
class QASample:
    question: str
    answer: str
    style: str = "default"


@dataclass
class TeacherSpec:
    model: str
    direction: str = "general"
    weight: float = 1.0
    topics: List[str] = field(default_factory=list)
    style: str = "teacher_general"
    question_prompt: str = ""
    answer_prompt: str = ""


@dataclass
class TeacherPlan:
    teachers: List[TeacherSpec] = field(default_factory=list)
    selection_strategy: str = "weighted-round-robin"
    student_model: str = ""
    graph_route: Dict[str, Any] = field(default_factory=dict)


class QADataset:
    def __init__(
        self,
        samples: Sequence[QASample],
        tokenizer,
        max_length: int,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        item = self.samples[index]
        return {
            "question": item.question,
            "answer": item.answer,
            "style": item.style,
        }

    def collate_fn(self, batch: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        prompts = [f"Question: {x['question']}\nAnswer:" for x in batch]
        answers = [f" {x['answer']}" for x in batch]
        full_texts = [p + a for p, a in zip(prompts, answers)]

        encoded_full = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded_prompt = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        labels = encoded_full["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        prompt_lengths = encoded_prompt["attention_mask"].sum(dim=1)
        for row, prompt_len in enumerate(prompt_lengths):
            labels[row, : int(prompt_len.item())] = -100

        return {
            "input_ids": encoded_full["input_ids"],
            "attention_mask": encoded_full["attention_mask"],
            "labels": labels,
            "question": [x["question"] for x in batch],
            "answer": [x["answer"] for x in batch],
            "style": [x["style"] for x in batch],
        }


def read_append_corpus(path: Path) -> List[QASample]:
    if not path.exists():
        raise FileNotFoundError(f"append corpus not found: {path}")

    samples: List[QASample] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                question = str(obj.get("question", "")).strip()
                answer = str(obj.get("answer", "")).strip()
                style = str(obj.get("style", "default")).strip() or "default"
                if question and answer:
                    samples.append(QASample(question=question, answer=answer, style=style))
    elif path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("JSON corpus must be a list of {question, answer, style?}")
        for item in obj:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            style = str(item.get("style", "default")).strip() or "default"
            if question and answer:
                samples.append(QASample(question=question, answer=answer, style=style))
    else:
        raise ValueError("append corpus must be .json or .jsonl")

    if not samples:
        raise ValueError("append corpus has no valid QA pairs")
    return samples


def ollama_chat(client, model: str, prompt: str, temperature: float, num_ctx: int) -> str:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_ctx": num_ctx},
    )
    return response["message"]["content"].strip()


class LlamaServerClient:
    """Minimal ollama-compatible adapter over llama-server's OpenAI-compatible
    /v1/chat/completions endpoint.  Lets the self-play generators talk to the
    mainline backend without any ollama dependency (v8.0 migration of the
    ollama-era fine-tuning bridge)."""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, model: str, messages: Sequence[Dict[str, str]],
             options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import json as _json
        import urllib.request

        payload: Dict[str, Any] = {"model": model, "messages": messages,
                                   "stream": False}
        if options:
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "num_ctx" in options:
                payload["n_ctx"] = options["num_ctx"]
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return {"message": {"content": data["choices"][0]["message"]["content"]}}


def build_self_play_samples(
    client,
    model: str,
    seed_topics: Sequence[str],
    num_pairs: int,
    temperature: float,
    num_ctx: int,
) -> List[QASample]:
    samples: List[QASample] = []
    for _ in range(num_pairs):
        topic = random.choice(seed_topics)
        q_prompt = (
            "Generate exactly one concise Chinese question. "
            f"Topic: {topic}. "
            "Output only the question."
        )
        question = ollama_chat(client, model, q_prompt, temperature=temperature, num_ctx=num_ctx)

        a_prompt = (
            "Answer the following question in Chinese with a consistent personal style. "
            "Keep answer 80-180 Chinese characters.\n"
            f"Question: {question}"
        )
        answer = ollama_chat(client, model, a_prompt, temperature=temperature, num_ctx=num_ctx)
        if question and answer:
            samples.append(QASample(question=question, answer=answer, style="self_play"))
    return samples


def build_multi_teacher_samples(
    client,
    teachers: Sequence[TeacherSpec],
    seed_topics: Sequence[str],
    num_pairs: int,
    temperature: float,
    num_ctx: int,
    teacher_plan: TeacherPlan | None = None,
) -> List[QASample]:
    samples: List[QASample] = []
    teacher_plan = teacher_plan or TeacherPlan(teachers=list(teachers))
    allocations = allocate_teacher_pairs(
        teachers,
        num_pairs,
        selection_strategy=teacher_plan.selection_strategy,
        graph_route=teacher_plan.graph_route,
    )
    for teacher, pair_count in zip(teachers, allocations):
        topics = resolve_teacher_topics(
            teacher,
            seed_topics,
            selection_strategy=teacher_plan.selection_strategy,
            graph_route=teacher_plan.graph_route,
        )
        if not topics:
            topics = ["科技", "教育", "历史", "艺术", "工程"]
        for _ in range(pair_count):
            topic = random.choice(topics)
            q_prompt = render_teacher_prompt(
                teacher.question_prompt,
                topic=topic,
                direction=teacher.direction,
                question="",
            )
            if not q_prompt:
                q_prompt = (
                    "Generate exactly one concise Chinese question. "
                    f"Direction: {teacher.direction}. "
                    f"Topic: {topic}. "
                    "Output only the question."
                )
            question = ollama_chat(client, teacher.model, q_prompt, temperature=temperature, num_ctx=num_ctx)

            a_prompt = render_teacher_prompt(
                teacher.answer_prompt,
                topic=topic,
                direction=teacher.direction,
                question=question,
            )
            if not a_prompt:
                a_prompt = (
                    "Answer the following question in Chinese with a consistent personal style. "
                    f"Direction: {teacher.direction}. "
                    "Keep answer 80-180 Chinese characters.\n"
                    f"Question: {question}"
                )
            answer = ollama_chat(client, teacher.model, a_prompt, temperature=temperature, num_ctx=num_ctx)
            if question and answer:
                samples.append(QASample(question=question, answer=answer, style=teacher.style))
    return samples


def compute_style_centroid(encoder, answers: Sequence[str]):
    import torch.nn.functional as F

    vectors = encoder.encode(list(answers), convert_to_tensor=True, normalize_embeddings=True)
    centroid = vectors.mean(dim=0)
    centroid = F.normalize(centroid, dim=0)
    return centroid


def safe_hf_model_name(preferred: str, fallback: str) -> str:
    if "/" in preferred or preferred.startswith("."):
        return preferred
    if preferred.startswith("hf:"):
        return preferred[3:]
    return fallback


def generate_answer_with_lm(
    model,
    tokenizer,
    question: str,
    device,
    max_new_tokens: int,
) -> str:
    import torch

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "Answer:" in generated:
        return generated.split("Answer:", 1)[-1].strip()
    return generated.strip()


def parse_seed_topics(text: str) -> List[str]:
    topics = [x.strip() for x in text.split(",") if x.strip()]
    return topics or ["科技", "教育", "历史", "艺术", "工程"]


def parse_target_modules(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").replace(";", ",").split(",") if item.strip()]


def parse_manifest_topics(value: Any) -> List[str]:
    if isinstance(value, str):
        return parse_seed_topics(value)
    if isinstance(value, list):
        topics: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                topics.append(text)
        return topics
    return []


def normalize_topic_list(values: Sequence[str], limit: int = 12) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for raw in values:
        topic = str(raw).strip()
        if not topic:
            continue
        key = topic.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(topic)
        if len(normalized) >= limit:
            break
    return normalized


def parse_graph_route(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    topics = normalize_topic_list(parse_manifest_topics(payload.get("topics", payload.get("routeTopics", []))))
    summary = str(payload.get("summary", payload.get("graphSummary", ""))).strip()
    if len(summary) > 240:
        summary = summary[:237].rstrip() + "..."
    direction = str(payload.get("direction", "")).strip()
    source = str(payload.get("source", "graph")).strip() or "graph"

    route: Dict[str, Any] = {}
    if topics:
        route["topics"] = topics
    if summary:
        route["summary"] = summary
    if direction:
        route["direction"] = direction
    if route:
        route["source"] = source
    return route


def render_teacher_prompt(template: str, *, topic: str, direction: str, question: str) -> str:
    rendered = str(template or "")
    rendered = rendered.replace("{topic}", topic)
    rendered = rendered.replace("{direction}", direction)
    rendered = rendered.replace("{question}", question)
    return rendered.strip()


def compute_teacher_route_weight(
    teacher: TeacherSpec,
    selection_strategy: str = "weighted-round-robin",
    graph_route: Dict[str, Any] | None = None,
) -> float:
    base_weight = max(0.0, float(teacher.weight))
    if selection_strategy != "graph-weighted":
        return base_weight

    route = graph_route or {}
    route_topics = normalize_topic_list(parse_manifest_topics(route.get("topics", [])))
    route_topic_set = {topic.casefold() for topic in route_topics}
    route_summary = str(route.get("summary", "")).strip().casefold()
    route_direction = str(route.get("direction", "")).strip().casefold()
    if not route_topic_set and not route_summary and not route_direction:
        return base_weight

    teacher_topics = normalize_topic_list(teacher.topics)
    exact_overlap = sum(1 for topic in teacher_topics if topic.casefold() in route_topic_set)
    summary_overlap = sum(1 for topic in teacher_topics if topic and topic.casefold() in route_summary)
    direction_bonus = 0.0
    teacher_direction = teacher.direction.strip().casefold()
    if teacher_direction:
        if route_direction and teacher_direction == route_direction:
            direction_bonus = 1.0
        elif teacher_direction in route_summary:
            direction_bonus = 0.5

    route_multiplier = 1.0 + (1.35 * exact_overlap) + (0.35 * summary_overlap) + (0.65 * direction_bonus)
    return base_weight * max(1.0, route_multiplier)


def resolve_teacher_topics(
    teacher: TeacherSpec,
    seed_topics: Sequence[str],
    selection_strategy: str = "weighted-round-robin",
    graph_route: Dict[str, Any] | None = None,
) -> List[str]:
    teacher_topics = normalize_topic_list(teacher.topics)
    seed_topic_list = normalize_topic_list(seed_topics)
    if selection_strategy != "graph-weighted":
        return teacher_topics or seed_topic_list

    route = graph_route or {}
    route_topics = normalize_topic_list(parse_manifest_topics(route.get("topics", [])))
    if not route_topics:
        return teacher_topics or seed_topic_list

    route_topic_set = {topic.casefold() for topic in route_topics}
    matched_topics = [topic for topic in teacher_topics if topic.casefold() in route_topic_set]
    if matched_topics:
        return matched_topics
    if teacher_topics:
        return teacher_topics
    return route_topics or seed_topic_list


def load_teacher_plan(path: Path | None) -> TeacherPlan:
    if path is None:
        return TeacherPlan()
    if not path.exists():
        raise FileNotFoundError(f"teacher manifest not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("teacher manifest must be a JSON object")

    raw_teachers = payload.get("teachers", [])
    if not isinstance(raw_teachers, list) or not raw_teachers:
        raise ValueError("teacher manifest must define a non-empty teachers array")

    teachers: List[TeacherSpec] = []
    for index, raw in enumerate(raw_teachers, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"teacher entry {index} must be an object")
        model = str(raw.get("model", "")).strip()
        if not model:
            raise ValueError(f"teacher entry {index} is missing model")
        direction = str(raw.get("direction", raw.get("role", "general"))).strip() or "general"
        weight = float(raw.get("weight", 1.0))
        if weight <= 0.0:
            raise ValueError(f"teacher entry {index} weight must be > 0")
        style = str(raw.get("style", f"teacher_{direction}")).strip() or "teacher_general"
        teachers.append(
            TeacherSpec(
                model=model,
                direction=direction,
                weight=weight,
                topics=parse_manifest_topics(raw.get("topics", [])),
                style=style,
                question_prompt=str(raw.get("questionPrompt", "")).strip(),
                answer_prompt=str(raw.get("answerPrompt", "")).strip(),
            )
        )

    return TeacherPlan(
        teachers=teachers,
        selection_strategy=str(payload.get("selectionStrategy", "weighted-round-robin")).strip() or "weighted-round-robin",
        student_model=str(payload.get("studentModel", "")).strip(),
        graph_route=parse_graph_route(payload.get("graphRoute", {})),
    )


def allocate_teacher_pairs(
    teachers: Sequence[TeacherSpec],
    total_pairs: int,
    selection_strategy: str = "weighted-round-robin",
    graph_route: Dict[str, Any] | None = None,
) -> List[int]:
    if total_pairs <= 0 or not teachers:
        return [0 for _ in teachers]

    weights = [compute_teacher_route_weight(teacher, selection_strategy, graph_route) for teacher in teachers]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        weights = [1.0 for _ in teachers]
        total_weight = float(len(teachers))

    raw_allocations = [(float(total_pairs) * weight) / total_weight for weight in weights]
    counts = [int(math.floor(value)) for value in raw_allocations]
    remainder = total_pairs - sum(counts)
    ranked = sorted(
        range(len(teachers)),
        key=lambda index: (raw_allocations[index] - counts[index], weights[index], -index),
        reverse=True,
    )
    for index in ranked[:remainder]:
        counts[index] += 1
    return counts


def summarize_corpus(samples: Sequence[QASample]) -> Dict[str, Any]:
    styles = Counter(sample.style or "default" for sample in samples)
    question_chars = [len(sample.question) for sample in samples]
    answer_chars = [len(sample.answer) for sample in samples]
    return {
        "count": len(samples),
        "styles": dict(sorted(styles.items())),
        "questionCharsAvg": round(sum(question_chars) / len(question_chars), 2) if question_chars else 0.0,
        "answerCharsAvg": round(sum(answer_chars) / len(answer_chars), 2) if answer_chars else 0.0,
        "questionCharsMax": max(question_chars) if question_chars else 0,
        "answerCharsMax": max(answer_chars) if answer_chars else 0,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.self_play_pairs < 0:
        raise ValueError("self-play-pairs must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be > 0")
    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.max_length <= 0:
        raise ValueError("max-length must be > 0")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be > 0")
    if args.log_every <= 0:
        raise ValueError("log-every must be > 0")
    if args.lr <= 0.0 or args.lr > 1.0:
        raise ValueError("lr must be in (0, 1]")
    if args.alpha < 0.0 or args.alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if args.beta < 0.0 or args.beta > 1.0:
        raise ValueError("beta must be in [0, 1]")
    if (args.alpha + args.beta) <= 0.0:
        raise ValueError("alpha + beta must be > 0")
    if args.warmup_ratio < 0.0 or args.warmup_ratio > 1.0:
        raise ValueError("warmup-ratio must be in [0, 1]")
    if args.weight_decay < 0.0 or args.weight_decay > 10.0:
        raise ValueError("weight-decay must be in [0, 10]")
    if args.lora_rank <= 0:
        raise ValueError("lora-rank must be > 0")
    if args.lora_alpha <= 0:
        raise ValueError("lora-alpha must be > 0")
    if args.lora_dropout < 0.0 or args.lora_dropout >= 1.0:
        raise ValueError("lora-dropout must be in [0, 1)")
    if args.teacher_manifest is not None and args.teacher_manifest.suffix.lower() != ".json":
        raise ValueError("teacher-manifest must be a .json file")


def build_execution_report(
    args: argparse.Namespace,
    append_samples: Sequence[QASample],
    train_model_name: str,
    teacher_plan: TeacherPlan | None = None,
) -> Dict[str, Any]:
    teacher_plan = teacher_plan or TeacherPlan()
    teacher_allocations = allocate_teacher_pairs(
        teacher_plan.teachers,
        args.self_play_pairs,
        selection_strategy=teacher_plan.selection_strategy,
        graph_route=teacher_plan.graph_route,
    )
    return {
        "mode": "dry-run" if args.dry_run else "train",
        "trainingMode": args.training_mode,
        "appendCorpus": str(args.append_corpus),
        "outputDir": str(args.output_dir),
        "reportPath": str(args.report_path) if args.report_path else "",
        "ollamaModel": args.ollama_model,
        "hfModelRequested": args.hf_model,
        "hfModelResolved": train_model_name,
        "seedTopics": parse_seed_topics(args.seed_topics),
        "appendSampleCount": len(append_samples),
        "selfPlayPairs": args.self_play_pairs,
        "requiresOllama": args.self_play_pairs > 0 and not (args.llama_server_url or os.environ.get("AI_LLAMACPP_BASE_URL", "")),
        "distillation": {
            "enabled": bool(teacher_plan.teachers),
            "selectionStrategy": teacher_plan.selection_strategy,
            "studentModel": teacher_plan.student_model,
            "graphRoute": teacher_plan.graph_route,
            "teacherCount": len(teacher_plan.teachers),
            "teachers": [
                {
                    "model": teacher.model,
                    "direction": teacher.direction,
                    "weight": teacher.weight,
                    "topics": teacher.topics,
                    "style": teacher.style,
                    "allocatedSelfPlayPairs": allocated,
                }
                for teacher, allocated in zip(teacher_plan.teachers, teacher_allocations)
            ],
        },
        "weights": {"alpha": args.alpha, "beta": args.beta},
        "training": {
            "epochs": args.epochs,
            "batchSize": args.batch_size,
            "gradientAccumulationSteps": args.gradient_accumulation_steps,
            "lr": args.lr,
            "weightDecay": args.weight_decay,
            "warmupRatio": args.warmup_ratio,
            "maxLength": args.max_length,
            "maxNewTokens": args.max_new_tokens,
            "device": args.device,
        },
        "lora": {
            "enabled": args.training_mode == "lora",
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targetModulesRequested": parse_target_modules(args.lora_target_modules),
            "saveMergedModel": args.save_merged_model,
        },
        "appendCorpusSummary": summarize_corpus(append_samples),
    }


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def auto_detect_lora_target_modules(model) -> List[str]:
    detected: List[str] = []
    seen = set()
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in COMMON_LORA_TARGET_MODULES and leaf not in seen:
            detected.append(leaf)
            seen.add(leaf)
    if detected:
        return detected

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) < 2:
            continue
        leaf = name.rsplit(".", 1)[-1]
        lowered = name.lower()
        if leaf == "lm_head":
            continue
        if leaf not in seen and any(token in lowered for token in ("attn", "proj", "mlp", "ffn", "gate", "query", "value")):
            detected.append(leaf)
            seen.add(leaf)
    return detected[:16]


def count_parameters(model) -> Dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        numel = int(parameter.numel())
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def build_adapter_manifest(
    base_model: str,
    adapter_dir: Path,
    target_modules: Sequence[str],
    merged_dir: Path | None,
) -> Dict[str, Any]:
    return {
        "baseModel": base_model,
        "adapterDir": str(adapter_dir),
        "targetModules": list(target_modules),
        "mergedModelDir": str(merged_dir) if merged_dir else "",
        "llamaCppWorkflow": [
            "Use the adapter directory as the LoRA artifact for PEFT-compatible inference.",
            "If llama.cpp deployment is needed, convert the merged model directory to GGUF or run your existing llama.cpp conversion toolchain on the merged model.",
            "Keep the adapter metadata together with the base HF model name to preserve repeatable deployment.",
        ],
    }


def save_training_artifacts(args: argparse.Namespace, report: Dict[str, Any], model, tokenizer, train_model_name: str, target_modules: Sequence[str]) -> None:
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    if args.training_mode == "lora":
        adapter_dir = final_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        report.setdefault("artifacts", {})["adapterDir"] = str(adapter_dir)

        merged_dir = None
        if args.save_merged_model:
            merged_dir = final_dir / "merged_model"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            report.setdefault("artifacts", {})["mergedModelDir"] = str(merged_dir)

        manifest_path = final_dir / "adapter_manifest.json"
        manifest = build_adapter_manifest(train_model_name, adapter_dir, target_modules, merged_dir)
        write_report(manifest_path, manifest)
        report.setdefault("artifacts", {})["adapterManifest"] = str(manifest_path)
    else:
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        report.setdefault("artifacts", {})["finalCheckpoint"] = str(final_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ollama-assisted Transformer fine-tuning with true LoRA adapter training")
    parser.add_argument("--append-corpus", type=Path, required=True, help="Path to added QA corpus (.json/.jsonl)")
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints/personality_ft"))
    parser.add_argument("--training-mode", type=str, default="lora", choices=["lora", "full"], help="Use PEFT LoRA adapters or full checkpoint tuning")
    parser.add_argument("--ollama-model", type=str, default="gpt-oss:20b", help="Any available Ollama model name")
    parser.add_argument("--llama-server-url", type=str, default="", help="llama-server base URL (OpenAI-compatible /v1); when set, self-play uses it instead of Ollama (v8.0)")
    parser.add_argument("--hf-model", type=str, default="gpt-oss:20b", help="Trainable HuggingFace CausalLM")
    parser.add_argument("--fallback-hf-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--self-play-pairs", type=int, default=128)
    parser.add_argument("--seed-topics", type=str, default="科技,哲学,工程,社会,科幻")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="")
    parser.add_argument("--save-merged-model", action="store_true", help="Merge LoRA weights back into the base model after training")
    parser.add_argument("--alpha", type=float, default=0.6, help="weight for semantic distance")
    parser.add_argument("--beta", type=float, default=0.4, help="weight for style distance")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate corpus and parameters without model/Ollama execution")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional JSON report output path")
    parser.add_argument("--teacher-manifest", type=Path, default=None, help="Optional JSON manifest describing weighted multi-teacher distillation inputs")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    append_samples = read_append_corpus(args.append_corpus)
    validate_args(args)
    teacher_plan = load_teacher_plan(args.teacher_manifest)

    requested_student_model = teacher_plan.student_model or args.hf_model
    train_model_name = safe_hf_model_name(requested_student_model, args.fallback_hf_model)
    report = build_execution_report(args, append_samples, train_model_name, teacher_plan)
    if args.report_path:
        write_report(args.report_path, report)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    if args.training_mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

    if args.device == "cpu" and torch.cuda.is_available():
        args.device = "cuda"
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    report["training"]["device"] = str(device)

    seed_topics = parse_seed_topics(args.seed_topics)
    self_play_samples: List[QASample] = []
    llama_server_url = args.llama_server_url or os.environ.get(
        "AI_LLAMACPP_BASE_URL", "")
    if args.self_play_pairs > 0:
        if llama_server_url:
            from ollama import Client as OllamaClient  # only when needed
            _unused_ollama = OllamaClient  # noqa: F841
            chat_client = LlamaServerClient(llama_server_url)  # type: ignore[assignment]
        else:
            from ollama import Client as OllamaClient
            chat_client = OllamaClient()
        ollama_client = chat_client
        if teacher_plan.teachers:
            self_play_samples = build_multi_teacher_samples(
                client=ollama_client,
                teachers=teacher_plan.teachers,
                seed_topics=seed_topics,
                num_pairs=args.self_play_pairs,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                teacher_plan=teacher_plan,
            )
        else:
            self_play_samples = build_self_play_samples(
                client=ollama_client,
                model=args.ollama_model,
                seed_topics=seed_topics,
                num_pairs=args.self_play_pairs,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
            )

    all_samples = append_samples + self_play_samples
    if len(all_samples) < 2:
        raise RuntimeError("not enough training samples after self-play + append corpus")

    print(f"ollama model: {args.ollama_model}")
    print(f"hf train model: {train_model_name} (requested: {args.hf_model})")
    print(f"training mode: {args.training_mode}")

    tokenizer = AutoTokenizer.from_pretrained(train_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(train_model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    target_modules: List[str] = []
    if args.training_mode == "lora":
        requested_targets = parse_target_modules(args.lora_target_modules)
        target_modules = requested_targets or auto_detect_lora_target_modules(model)
        if not target_modules:
            raise RuntimeError("could not auto-detect LoRA target modules; please pass --lora-target-modules")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        report["lora"]["targetModulesResolved"] = target_modules

    model.to(device)
    model.train()
    report["parameterCounts"] = count_parameters(model)

    dataset = QADataset(all_samples, tokenizer=tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=str(device))
    style_centroid = compute_style_centroid(encoder, [sample.answer for sample in append_samples]).to(device)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(dataloader) / args.gradient_accumulation_steps))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ce_loss = outputs.loss

            semantic_distances: List[float] = []
            style_distances: List[float] = []
            for question, ref_answer in zip(batch["question"], batch["answer"]):
                generated_answer = generate_answer_with_lm(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )

                pair_vectors = encoder.encode(
                    [generated_answer, ref_answer],
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                ).to(device)
                semantic_sim = F.cosine_similarity(pair_vectors[0], pair_vectors[1], dim=0).clamp(-1.0, 1.0)
                style_sim = F.cosine_similarity(pair_vectors[0], style_centroid, dim=0).clamp(-1.0, 1.0)

                semantic_distances.append(float((1.0 - semantic_sim).item()))
                style_distances.append(float((1.0 - style_sim).item()))

            semantic_loss = torch.tensor(semantic_distances, dtype=torch.float32, device=device).mean()
            style_loss = torch.tensor(style_distances, dtype=torch.float32, device=device).mean()

            interpolation_weight = args.alpha * semantic_loss + args.beta * style_loss
            total_loss = ce_loss * (1.0 + interpolation_weight)
            scaled_loss = total_loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            if batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0 or global_step == 1:
                    print(
                        f"[step {global_step}] "
                        f"ce={ce_loss.item():.4f} sem={semantic_loss.item():.4f} "
                        f"style={style_loss.item():.4f} total={total_loss.item():.4f}"
                    )

        epoch_dir = args.output_dir / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

    report["generatedSelfPlaySamples"] = len(self_play_samples)
    report["totalTrainingSamples"] = len(all_samples)
    save_training_artifacts(args, report, model, tokenizer, train_model_name, target_modules)
    if args.report_path:
        write_report(args.report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()