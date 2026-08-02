import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "transformer_ollama_fine_tuning.py"
SPEC = importlib.util.spec_from_file_location("phoenix_transformer_ollama_fine_tuning", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def chat(self, model: str, messages: list[dict], options: dict) -> dict:
        prompt = messages[0]["content"]
        self.calls.append((model, prompt))
        if "Output only the question." in prompt:
            return {"message": {"content": f"{model} question"}}
        return {"message": {"content": f"{model} answer"}}


class TransformerOllamaFineTuningTests(unittest.TestCase):
    def test_load_teacher_plan_reads_weighted_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = pathlib.Path(tmpdir) / "teachers.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "studentModel": "hf:Qwen/Qwen2.5-1.5B-Instruct",
                        "selectionStrategy": "weighted-round-robin",
                        "teachers": [
                            {"model": "teacher-reasoner", "direction": "reasoning", "weight": 2.0, "topics": ["工程", "诊断"]},
                            {"model": "teacher-style", "role": "style", "weight": 1.0, "topics": "文学,表达"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            plan = MODULE.load_teacher_plan(manifest_path)

        self.assertEqual(plan.student_model, "hf:Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(plan.selection_strategy, "weighted-round-robin")
        self.assertEqual(len(plan.teachers), 2)
        self.assertEqual(plan.teachers[0].topics, ["工程", "诊断"])
        self.assertEqual(plan.teachers[1].direction, "style")
        self.assertEqual(plan.teachers[1].topics, ["文学", "表达"])

    def test_load_teacher_plan_reads_graph_route_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = pathlib.Path(tmpdir) / "teachers_graph.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "studentModel": "hf:Qwen/Qwen2.5-1.5B-Instruct",
                        "selectionStrategy": "graph-weighted",
                        "graphRoute": {
                            "topics": ["工程", "诊断", "工程"],
                            "summary": "damaged solar array diagnostics require a careful engineering route" * 8,
                            "direction": "reasoning",
                        },
                        "teachers": [
                            {"model": "teacher-reasoner", "direction": "reasoning", "weight": 1.0, "topics": ["工程", "诊断"]},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            plan = MODULE.load_teacher_plan(manifest_path)

        self.assertEqual(plan.selection_strategy, "graph-weighted")
        self.assertEqual(plan.graph_route["topics"], ["工程", "诊断"])
        self.assertEqual(plan.graph_route["direction"], "reasoning")
        self.assertLessEqual(len(plan.graph_route["summary"]), 240)

    def test_allocate_teacher_pairs_preserves_total(self) -> None:
        teachers = [
            MODULE.TeacherSpec(model="teacher-a", direction="reasoning", weight=3.0),
            MODULE.TeacherSpec(model="teacher-b", direction="style", weight=1.0),
            MODULE.TeacherSpec(model="teacher-c", direction="safety", weight=2.0),
        ]

        allocation = MODULE.allocate_teacher_pairs(teachers, 12)

        self.assertEqual(sum(allocation), 12)
        self.assertGreater(allocation[0], allocation[1])
        self.assertGreater(allocation[2], allocation[1])

    def test_allocate_teacher_pairs_biases_graph_weighted_route(self) -> None:
        teachers = [
            MODULE.TeacherSpec(model="teacher-reasoner", direction="reasoning", weight=1.0, topics=["工程", "诊断"]),
            MODULE.TeacherSpec(model="teacher-style", direction="style", weight=1.0, topics=["文学", "表达"]),
        ]

        allocation = MODULE.allocate_teacher_pairs(
            teachers,
            6,
            selection_strategy="graph-weighted",
            graph_route={"topics": ["工程", "诊断"], "summary": "engineering diagnostics need reasoning", "direction": "reasoning"},
        )

        self.assertEqual(sum(allocation), 6)
        self.assertGreater(allocation[0], allocation[1])

    def test_build_execution_report_includes_distillation_plan(self) -> None:
        append_samples = [MODULE.QASample(question="q", answer="a")]
        args = MODULE.build_arg_parser().parse_args([
            "--append-corpus",
            "dummy.jsonl",
            "--self-play-pairs",
            "6",
            "--dry-run",
        ])
        plan = MODULE.TeacherPlan(
            teachers=[
                MODULE.TeacherSpec(model="teacher-a", direction="reasoning", weight=2.0, topics=["工程"]),
                MODULE.TeacherSpec(model="teacher-b", direction="style", weight=1.0, topics=["表达"]),
            ],
            selection_strategy="weighted-round-robin",
            student_model="hf:Qwen/Qwen2.5-1.5B-Instruct",
        )

        report = MODULE.build_execution_report(args, append_samples, "Qwen/Qwen2.5-1.5B-Instruct", plan)

        self.assertTrue(report["distillation"]["enabled"])
        self.assertEqual(report["distillation"]["teacherCount"], 2)
        self.assertEqual(report["distillation"]["studentModel"], "hf:Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(report["distillation"]["teachers"][0]["allocatedSelfPlayPairs"], 4)
        self.assertEqual(report["distillation"]["teachers"][1]["allocatedSelfPlayPairs"], 2)

    def test_build_execution_report_includes_graph_route(self) -> None:
        append_samples = [MODULE.QASample(question="q", answer="a")]
        args = MODULE.build_arg_parser().parse_args([
            "--append-corpus",
            "dummy.jsonl",
            "--self-play-pairs",
            "4",
            "--dry-run",
        ])
        plan = MODULE.TeacherPlan(
            teachers=[MODULE.TeacherSpec(model="teacher-a", direction="reasoning", weight=1.0, topics=["工程"])],
            selection_strategy="graph-weighted",
            graph_route={"topics": ["工程"], "summary": "engineering diagnostics", "source": "session-graph"},
        )

        report = MODULE.build_execution_report(args, append_samples, "Qwen/Qwen2.5-1.5B-Instruct", plan)

        self.assertEqual(report["distillation"]["graphRoute"]["topics"], ["工程"])
        self.assertEqual(report["distillation"]["graphRoute"]["source"], "session-graph")

    def test_build_multi_teacher_samples_uses_weighted_models(self) -> None:
        client = FakeOllamaClient()
        teachers = [
            MODULE.TeacherSpec(model="teacher-a", direction="reasoning", weight=2.0, topics=["工程"], style="teacher_reasoning"),
            MODULE.TeacherSpec(model="teacher-b", direction="style", weight=1.0, topics=["表达"], style="teacher_style"),
        ]

        samples = MODULE.build_multi_teacher_samples(
            client=client,
            teachers=teachers,
            seed_topics=["默认主题"],
            num_pairs=3,
            temperature=0.3,
            num_ctx=2048,
        )

        self.assertEqual(len(samples), 3)
        styles = [sample.style for sample in samples]
        self.assertEqual(styles.count("teacher_reasoning"), 2)
        self.assertEqual(styles.count("teacher_style"), 1)
        models = [model for model, _prompt in client.calls]
        self.assertEqual(models.count("teacher-a"), 4)
        self.assertEqual(models.count("teacher-b"), 2)

    def test_build_multi_teacher_samples_prefers_graph_aligned_teacher(self) -> None:
        client = FakeOllamaClient()
        teachers = [
            MODULE.TeacherSpec(model="teacher-reasoner", direction="reasoning", weight=1.0, topics=["工程", "诊断"], style="teacher_reasoning"),
            MODULE.TeacherSpec(model="teacher-style", direction="style", weight=1.0, topics=["文学", "表达"], style="teacher_style"),
        ]
        plan = MODULE.TeacherPlan(
            teachers=teachers,
            selection_strategy="graph-weighted",
            graph_route={"topics": ["工程"], "summary": "engineering route diagnostics", "direction": "reasoning"},
        )

        samples = MODULE.build_multi_teacher_samples(
            client=client,
            teachers=teachers,
            seed_topics=["默认主题"],
            num_pairs=4,
            temperature=0.3,
            num_ctx=2048,
            teacher_plan=plan,
        )

        self.assertEqual(len(samples), 4)
        styles = [sample.style for sample in samples]
        self.assertGreater(styles.count("teacher_reasoning"), styles.count("teacher_style"))

    def test_parse_graph_route_tolerates_malformed_payload(self) -> None:
        route = MODULE.parse_graph_route({"topics": ["工程", None, 7, "工程"], "summary": 42, "source": "api"})

        self.assertEqual(route["topics"], ["工程", "None", "7"])
        self.assertEqual(route["summary"], "42")
        self.assertEqual(route["source"], "api")


if __name__ == "__main__":
    unittest.main()