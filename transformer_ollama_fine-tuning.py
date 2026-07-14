#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from ollama import Client as OllamaClient
from sentence_transformers import SentenceTransformer


@dataclass
class QASample:
    question: str
    answer: str
    style: str = "default"


class QADataset(Dataset):
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

    def collate_fn(self, batch: Sequence[Dict[str, str]]) -> Dict[str, torch.Tensor | List[str]]:
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


def ollama_chat(client: OllamaClient, model: str, prompt: str, temperature: float, num_ctx: int) -> str:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_ctx": num_ctx},
    )
    return response["message"]["content"].strip()


def build_self_play_samples(
    client: OllamaClient,
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


def compute_style_centroid(encoder: SentenceTransformer, answers: Sequence[str]) -> torch.Tensor:
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
    device: torch.device,
    max_new_tokens: int,
) -> str:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ollama-assisted Transformer fine-tuning with semantic/style interpolation")
    parser.add_argument("--append-corpus", type=Path, required=True, help="Path to added QA corpus (.json/.jsonl)")
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints/personality_ft"))
    parser.add_argument("--ollama-model", type=str, default="gpt-oss:20b", help="Any available Ollama model name")
    parser.add_argument("--hf-model", type=str, default="gpt-oss:20b", help="Trainable HuggingFace CausalLM")
    parser.add_argument("--fallback-hf-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--self-play-pairs", type=int, default=128)
    parser.add_argument("--seed-topics", type=str, default="科技,哲学,工程,社会,科幻")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.6, help="weight for semantic distance")
    parser.add_argument("--beta", type=float, default=0.4, help="weight for style distance")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=20)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    append_samples = read_append_corpus(args.append_corpus)

    ollama_client = OllamaClient()
    seed_topics = parse_seed_topics(args.seed_topics)
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

    train_model_name = safe_hf_model_name(args.hf_model, args.fallback_hf_model)
    print(f"ollama model: {args.ollama_model}")
    print(f"hf train model: {train_model_name} (requested: {args.hf_model})")

    tokenizer = AutoTokenizer.from_pretrained(train_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(train_model_name)
    model.to(device)
    model.train()

    dataset = QADataset(all_samples, tokenizer=tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=str(device))
    style_centroid = compute_style_centroid(encoder, [s.answer for s in append_samples]).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(dataloader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    global_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
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

                gen_vec = pair_vectors[0]
                style_sim = F.cosine_similarity(gen_vec, style_centroid, dim=0).clamp(-1.0, 1.0)

                semantic_distances.append(float((1.0 - semantic_sim).item()))
                style_distances.append(float((1.0 - style_sim).item()))

            semantic_loss = torch.tensor(semantic_distances, dtype=torch.float32, device=device).mean()
            style_loss = torch.tensor(style_distances, dtype=torch.float32, device=device).mean()

            interpolation_weight = args.alpha * semantic_loss + args.beta * style_loss
            total_loss = ce_loss * (1.0 + interpolation_weight)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

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
        print(f"saved checkpoint: {epoch_dir}")

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"training done, final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
