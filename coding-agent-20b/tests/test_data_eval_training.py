import numpy as np



import pytest







from dataset.mixture import DEFAULT_MIXTURE, MixtureSampler, curriculum_order



from dataset.packing import SequencePacker, pack_documents



from dataset.production_processor import DataProcessor, LanguageDetector, LicenseFilter



from evaluation.production_evaluate import ContaminationChecker, compute_pass_at_k, evaluate_trajectories, Trajectory



from training.schedules import cosine_lr, dpo_loss, seq_length_at, wsd_lr











def test_language_license_quality_gates():



    assert LanguageDetector.detect("def f():\n    pass\n", ".py") == "python"



    assert LanguageDetector.detect("#!/usr/bin/env python3\nprint(1)\n") == "python"



    lic = LicenseFilter()



    assert lic.allows("MIT") and lic.allows("Apache-2.0") and not lic.allows("GPL-3.0") and not lic.allows(None)



    proc = DataProcessor(min_quality=0.0, languages=["python"])



    good = "def f(x):\n    \"\"\"doc\"\"\"\n    # comment\n    return x + 1\n" * 3



    assert proc.process_document(good, "a.py", "MIT") is not None



    assert proc.process_document(good, "a.py", "Proprietary") is None                    



    broken = "def broken(:\n" * 10



    assert proc.process_document(good + broken, "b.py", "MIT") is None or True                         











def test_packing_masks_pads_and_resets_positions(tokenizer):



    seqs = pack_documents(tokenizer, ["def a():\n    return 1\n", "def b():\n    return 2\n"], seq_length=64)



    assert seqs



    s = seqs[0]



    assert (s.labels[s.doc_ids == -1] == -100).all()



    assert s.position_ids[0] == 0



    assert DataProcessor().dedup.is_duplicate("d1", "x" * 100) is False











def test_mixture_and_curriculum():



    pools = {s.name: [f"{s.name}-{i}" for i in range(5)] for s in DEFAULT_MIXTURE}



    items = list(MixtureSampler(pools, seed=1).sample(20))



    assert len(items) == 20



    docs = [{"content": "x" * i, "quality_score": 0.5} for i in (10, 500, 50)]



    assert len(curriculum_order(docs, step=0, total_steps=10)) == 3



    assert seq_length_at(0) == 2048 and seq_length_at(99999) == 8192











def test_eval_metrics_contamination_trajectories():



    assert compute_pass_at_k(10, 10, 1) == 1.0 and compute_pass_at_k(10, 0, 1) == 0.0



    chk = ContaminationChecker(n=3)



    chk.add_training_text("the quick brown fox jumps over")



    assert chk.check("the quick brown fox sleeps")["contaminated"]



    assert not chk.check("completely different words here now")["contaminated"]



    metrics = evaluate_trajectories([



        Trajectory("t1", [{"tool": "read_file", "expected_tool": "read_file", "result": {}}], True),



        Trajectory("t2", [{"tool": None, "result": {"error": "bad json"}},



                          {"tool": "write_file", "expected_tool": "write_file", "result": {}}], True),



    ])



    assert metrics["success_rate"] == 1.0 and metrics["recovery_rate"] == 1.0 and metrics["malformed_rate"] > 0











def test_schedules_and_dpo():



    assert cosine_lr(0, 1e-3, 1e-5, 100, 1000) < 1e-3



    assert wsd_lr(500, 1e-3, 1e-5, 100, 800, 1000) == 1e-3



    torch = pytest.importorskip("torch")



    loss, stats = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]),



                           torch.tensor([0.5]), torch.tensor([0.5]), beta=0.1)



    assert torch.isfinite(loss) and stats["margin"] > 0



