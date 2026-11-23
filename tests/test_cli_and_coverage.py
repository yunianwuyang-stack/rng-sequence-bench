"""Additional CLI, ragged-reader, and recovery-coverage tests."""

import json
import sys

import numpy as np
import pytest

from rng_sequence_bench import checks as checks_mod
from rng_sequence_bench import diagnostics as diag_mod
from rng_sequence_bench import recovery
from rng_sequence_bench.__main__ import main as cli_main
from rng_sequence_bench.data import from_ragged, read_ragged_trials, read_trials


def test_sequence_checks_cli_forecast_and_sequence_npz(tmp_path, monkeypatch):
    npz_path = tmp_path / "inputs.npz"
    out_path = tmp_path / "diag.json"
    rng = np.random.default_rng(101)
    choices = rng.integers(0, 3, size=(4, 15))
    probs = np.full((4, 14, 3), 1 / 3)
    np.savez(
        npz_path,
        probabilities=probs,
        targets=choices[:, 1:],
        choices=choices,
        coordinates=np.array([[0.0], [1.0], [2.0]]),
        reference=np.array([1 / 3, 1 / 3, 1 / 3]),
    )
    monkeypatch.setattr(sys, "argv", ["rng-sequence-checks", str(npz_path), str(out_path)])
    checks_mod.main()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(report) == {"forecasts", "sequences"}

    bad_npz = tmp_path / "bad.npz"
    np.savez(bad_npz, probabilities=probs)
    monkeypatch.setattr(sys, "argv", ["rng-sequence-checks", str(bad_npz), str(out_path)])
    with pytest.raises(SystemExit):
        checks_mod.main()

    empty_npz = tmp_path / "empty.npz"
    np.savez(empty_npz, unrelated=np.array([1, 2]))
    monkeypatch.setattr(sys, "argv", ["rng-sequence-checks", str(empty_npz), str(out_path)])
    with pytest.raises(SystemExit):
        checks_mod.main()

    bad_seq_npz = tmp_path / "bad_seq.npz"
    np.savez(bad_seq_npz, choices=choices)
    monkeypatch.setattr(sys, "argv", ["rng-sequence-checks", str(bad_seq_npz), str(out_path)])
    with pytest.raises(SystemExit):
        checks_mod.main()


def test_cli_main_inprocess_standard_and_standardized(tmp_path, monkeypatch):
    csv_path = tmp_path / "trials.csv"
    csv_path.write_text(
        "subject,trial,choice\n"
        + "".join(f"{i},{t + 1},{(i + t) % 3}\n" for i in range(4) for t in range(18)),
        encoding="utf-8",
    )
    coords_path = tmp_path / "coords.json"
    coords_path.write_text("[[0], [1], [2]]", encoding="utf-8")

    out1 = tmp_path / "std_off.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rng-sequence-bench",
            str(csv_path),
            str(out1),
            "--alphabet",
            "0,1,2",
            "--coordinates",
            str(coords_path),
            "--origins",
            "10,14",
            "--width",
            "4",
            "--permutations",
            "5",
            "--bootstrap",
            "5",
            "--simulations",
            "2",
            "--frequency-window",
            "3",
            "--include-pooled",
        ],
    )
    cli_main()
    rep1 = json.loads(out1.read_text(encoding="utf-8"))
    assert rep1["coordinate_kind"] == "supplied"
    assert "pooled_RDC" in rep1["prediction"]["participant_bootstrap"]

    out2 = tmp_path / "std_on.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rng-sequence-bench",
            str(csv_path),
            str(out2),
            "--alphabet",
            "0,1,2",
            "--origins",
            "12,15",
            "--width",
            "3",
            "--standardized",
            "--inner-width",
            "4",
            "--permutations",
            "5",
        ],
    )
    cli_main()
    rep2 = json.loads(out2.read_text(encoding="utf-8"))
    assert rep2["coordinate_kind"] == "ordinal"


def test_chain_diagnostics_cli_validation_and_mock_rscript(tmp_path, monkeypatch):
    draws = tmp_path / "draws.csv"
    draws.write_text("chain,draw,divergent__,treedepth__,energy__,mu\n1,1,0,2,10.0,0.1\n", encoding="utf-8")
    out_dir = tmp_path / "diag_out"

    monkeypatch.setattr(
        sys,
        "argv",
        ["rng-chain-diagnostics", str(draws), str(out_dir), "--max-treedepth", "0"],
    )
    with pytest.raises(SystemExit):
        diag_mod.main()

    monkeypatch.setattr(diag_mod.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rng-chain-diagnostics", str(draws), str(out_dir), "--max-treedepth", "10"],
    )
    with pytest.raises(SystemExit):
        diag_mod.main()

    calls = []
    monkeypatch.setattr(diag_mod.shutil, "which", lambda _: "/usr/bin/Rscript")
    monkeypatch.setattr(diag_mod.subprocess, "run", lambda cmd, check: calls.append((cmd, check)))
    diag_mod.main()
    assert len(calls) == 1 and calls[0][1] is True and calls[0][0][-1] == "10"


def test_data_readers_validation_edges(tmp_path):
    with pytest.raises(ValueError):
        from_ragged([])
    with pytest.raises(ValueError):
        from_ragged([[]])
    with pytest.raises(ValueError):
        from_ragged([[-1, -1]])

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("subject,choice\na,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_trials(bad_csv, ["0", "1"])
    with pytest.raises(ValueError):
        read_ragged_trials(bad_csv, ["0", "1"])

    bad_choice = tmp_path / "bad_choice.csv"
    bad_choice.write_text("subject,trial,choice\na,1,9\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_trials(bad_choice, ["0", "1"])
    with pytest.raises(ValueError):
        read_ragged_trials(bad_choice, ["0", "1"])

    empty_subj = tmp_path / "empty.csv"
    empty_subj.write_text("subject,trial,choice\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_trials(empty_subj, ["0", "1"])
    with pytest.raises(ValueError):
        read_ragged_trials(empty_subj, ["0", "1"])


def test_recovery_benchmark_small_and_validation():
    with pytest.raises(ValueError):
        recovery.recovery_benchmark(sizes=(2,), lengths=(12,), strengths=(0.65,))
    progress_calls = []
    res = recovery.recovery_benchmark(
        sizes=(4,),
        lengths=(12,),
        strengths=(0.65,),
        replicates=1,
        workers=1,
        progress=lambda done, total: progress_calls.append((done, total)),
    )
    assert len(res["rows"]) == len(recovery.GENERATORS)
    assert res["seed"] == 20261211
