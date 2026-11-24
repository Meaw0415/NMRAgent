import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_benchmark as runner


class BenchmarkTests(unittest.TestCase):
    def test_reference_and_dataset_prompt_do_not_reach_agent(self):
        row = runner.normalize_record({"idx": 4, "prompt": "answer is CCO", "answer": "CCO", "extra_info": {"formula": "C2H6O", "h_shifts": [1, 1, 1], "c_shifts": [20]}}, 0, "nmrgym")
        query = runner.agent_input(row)
        self.assertNotIn("CCO", json.dumps(query))
        self.assertEqual(query["h_shifts"], [1, 1, 1])
        self.assertEqual(set(query), {"task", "formula", "h_shifts", "c_shifts"})

    def test_schema_and_missing_spectrum(self):
        row = runner.normalize_record({"formula": "CH4", "c_shift": [1], "smiles": "C"}, 9, "nmrshiftdb")
        self.assertEqual(row["idx"], 9)
        self.assertEqual(row["h_shifts"], [])
        with self.assertRaises(ValueError):
            runner.normalize_record({"formula": "CH4"}, 0, "exp450")

    def test_exact_accuracy_is_not_fingerprint_identity_or_candidate_recall(self):
        rows = [
            {"idx": 0, "gt_smiles": "F[C@H](Cl)Br", "pred_smiles": "F[C@@H](Cl)Br", "verifier_candidates": []},
            {"idx": 1, "gt_smiles": "CCO", "pred_smiles": "", "error": "failed", "verifier_candidates": [{"smiles": "CCO"}]},
        ]
        result = runner.score_records(rows)
        self.assertEqual(result["final_answer_top1_accuracy"], 0.5)
        self.assertEqual(result["verifier_candidate_topk_accuracy"]["1"], 0.5)
        with self.assertRaises(ValueError):
            runner.score_records(rows + [rows[0]])

    def test_full_prompt_trace_uses_actual_request_messages(self):
        from agents.multi_agent_nmr_v2 import MultiAgentNMRV2
        agent = MultiAgentNMRV2.__new__(MultiAgentNMRV2)
        agent.trace_full_prompts = True
        agent.model_name_or_path = "test"
        agent.backend_kwargs = {}
        logged, calls = [], []
        agent._trace = lambda event, payload: logged.append((event, payload))
        content = json.dumps({"analysis": "a" * 5000})
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")])
        agent._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        agent._invoke_json_llm("planner", "actual system prompt", {"formula": "CH4"})
        self.assertEqual(logged[0][1]["messages"], calls[0]["messages"])
        self.assertEqual(logged[-1][1]["content"], content)

    def test_trace_omits_api_credentials_and_keeps_exact_rerank_arguments(self):
        from agents.multi_agent_nmr_v2 import MultiAgentNMRV2
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            agent = MultiAgentNMRV2(model_name_or_path="test", api_key="test-secret-do-not-log",
                                   trace_log_path=str(trace), trace_full_prompts=True)
            captured = []
            def rerank(**kwargs):
                captured.append(kwargs)
                return {"candidates": [{"smiles": "CCO"}], "count": 1}
            agent._tool_map = lambda names: {"nmr_rerank": rerank}
            agent._invoke_json_llm = lambda *args: {"verdict": "accept", "top_candidate": "CCO"}
            agent._verifier_node({"formula": "C2H6O", "h_shifts": [1], "c_shifts": [20], "candidate_buffer": [{"smiles": "CCO"}]})
            text = trace.read_text()
            self.assertNotIn("test-secret-do-not-log", text)
            records = [json.loads(line) for line in text.splitlines()]
            logged = next(r for r in records if r["event"] == "tool_start")
            self.assertEqual(logged["args"], captured[0])

    def test_batch_failure_and_resume_keep_one_record_per_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "input.json"
            source.write_text(json.dumps([{"idx": 7, "formula": "C2H6O", "h_shifts": [1], "answer": "CCO"}]))
            argv = ["run_benchmark.py", "--benchmark", "nmrgym", "--input", str(source), "--output-dir", str(base / "out")]
            from agents.multi_agent_nmr_v2 import MultiAgentNMRV2
            with patch.object(sys, "argv", argv), patch.object(MultiAgentNMRV2, "from_config", side_effect=RuntimeError("test failure")):
                runner.main()
            rows = runner.load_records(base / "out" / "predictions.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(runner.score_records(rows)["final_answer_top1_accuracy"], 0)
            with patch.object(sys, "argv", argv + ["--resume"]), patch.object(MultiAgentNMRV2, "from_config") as build:
                runner.main()
                build.assert_not_called()
            self.assertEqual(len(runner.load_records(base / "out" / "predictions.jsonl")), 1)

    def test_successful_batch_saves_state_and_only_passes_spectral_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "input.json"
            source.write_text(json.dumps([{"idx": 3, "formula": "C2H6O", "c_shifts": [20, 60], "reference_smiles": "CCO"}]))
            calls = []
            def run(**query):
                calls.append(query)
                return {"final_answer": "CCO", "cand_list": [{"smiles": "CCO"}]}
            from agents.multi_agent_nmr_v2 import MultiAgentNMRV2
            argv = ["run_benchmark.py", "--benchmark", "exp450", "--input", str(source), "--output-dir", str(base / "out")]
            with patch.object(sys, "argv", argv), patch.object(MultiAgentNMRV2, "from_config", return_value=SimpleNamespace(run=run)):
                runner.main()
            self.assertEqual(set(calls[0]), {"task", "formula", "h_shifts", "c_shifts"})
            self.assertNotIn("CCO", json.dumps(calls[0]))
            self.assertTrue((base / "out/cases/sample_000000/state.json").is_file())
            summary = json.loads((base / "out/summary.json").read_text())
            self.assertEqual(summary["final_answer_top1_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
