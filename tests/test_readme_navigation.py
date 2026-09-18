from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
PRIMARY_GUIDES = (
    "INSTALL.md",
    "THIRD_PARTY.md",
    "docs/BENCHMARK.md",
    "docs/CODE_STRUCTURE.md",
    "docs/CONFIGURATION.md",
    "docs/CUSTOM_MODELS.md",
    "docs/EVALUATION.md",
    "docs/VIDEO_MODELS.md",
    "examples/quickstart/README.md",
    "examples/quickstart/VLM.md",
)
REFERENCE_GUIDES = (
    "configs/README.md",
    "docs/MODEL_ASSETS.md",
    "examples/custom_models/README.md",
    "integrations/README.md",
)


def _text(relative: str) -> str:
    return (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")


def test_readme_has_one_getting_started_path_then_goal_branches() -> None:
    text = README.read_text(encoding="utf-8")

    for relative in (*PRIMARY_GUIDES, *REFERENCE_GUIDES):
        assert (REPOSITORY_ROOT / relative).is_file(), relative

    task_suite = text.index("## 🧪 Benchmark task suite")
    getting_started = text.index("## 🚀 Get started")
    choose_goal = text.index("## 🧭 Choose your next goal")
    configuration = text.index("## 🗂️ Configuration at a glance")
    citation = text.index("## 📌 Citation")
    assert task_suite < getting_started < choose_goal < configuration < citation

    start_text = text[getting_started:choose_goal]
    assert "](INSTALL.md" in start_text
    assert "](examples/quickstart/README.md" in start_text
    assert "### 1. Install once" in start_text
    assert "### 2. Run the bundled case" in start_text

    goal_text = text[choose_goal:configuration]
    assert "| Goal | Additional requirement | Continue with |" in goal_text
    for relative in (
        "docs/BENCHMARK.md",
        "docs/VIDEO_MODELS.md",
        "docs/EVALUATION.md",
        "examples/quickstart/VLM.md",
        "docs/CUSTOM_MODELS.md",
        "docs/CODE_STRUCTURE.md",
    ):
        assert f"]({relative}" in goal_text

    assert "## 🔁 Run the full benchmark" not in text
    assert "## 🎥 Evaluate your video generator or WAM" not in text
    assert "## 📊 Evaluate saved results" not in text
    assert "## 👁️ VLM evaluation" not in text
    assert "## 🔌 Replace model backends" not in text
    assert "<DREAM_EXE_BENCHMARK_REPOSITORY>" not in text


def test_root_readme_owns_the_one_case_entry_command() -> None:
    entry = (
        "--workspace examples/quickstart/workspace.json \\\n"
        "  --spec examples/quickstart/run.json"
    )
    root_text = README.read_text(encoding="utf-8")
    assert root_text.count(entry) == 1

    for relative in (
        "INSTALL.md",
        "examples/quickstart/README.md",
        "docs/BENCHMARK.md",
    ):
        assert entry not in _text(relative), relative


def test_install_owns_every_required_first_run_acquisition() -> None:
    install = _text("INSTALL.md")
    readme = README.read_text(encoding="utf-8")

    required_steps = (
        "git lfs pull",
        '.[assets,video,sim,robocasa-runtime,dvd-runtime,tracking-runtime,region-runtime]',
        "integrations/setup.py core",
        "--accept-noncommercial-licenses",
        "hf download kaimingyang/DVD_for_Dream.exe",
        'DVD/lora/specific/rc_cheesybread_ep000001/*',
    )
    for step in required_steps:
        assert step in install
        assert step not in readme

    assert "Full benchmark data" in install
    assert "Alternative models and hosted evaluation" in install
    for optional in ("VDA", "FoundationPose", "Wan2.2", "VLM"):
        assert optional in install


def test_install_guide_explains_provider_patch_command() -> None:
    install = _text("INSTALL.md")

    assert "python integrations/setup.py core" in install
    assert "Do not run `git apply` manually" in install
    for provider in ("dvd", "grounding_dino", "robocasa"):
        manifest = f"integrations/patches/{provider}/manifest.json"
        assert manifest in install
        assert (REPOSITORY_ROOT / manifest).is_file()


def test_full_benchmark_acquisition_has_one_workflow_owner() -> None:
    benchmark = _text("docs/BENCHMARK.md")
    assert "hf download kaimingyang/Dream.exe" in benchmark
    assert "https://huggingface.co/datasets/kaimingyang/Dream.exe" in benchmark
    assert "hf download kaimingyang/DVD_for_Dream.exe" in benchmark
    assert "--local-dir data" in benchmark
    assert "`roots.bench`" in benchmark
    assert "`roots.published_results`" in benchmark
    assert "'results/runs/**' 'results/videos/**' 'results/experiments/**'" in benchmark

    for relative in (
        "README.md",
        "INSTALL.md",
        "configs/README.md",
        "docs/CONFIGURATION.md",
        "examples/quickstart/README.md",
    ):
        assert "<DREAM_EXE_BENCHMARK_REPOSITORY>" not in _text(relative), relative


def test_optional_model_asset_commands_stay_in_model_reference() -> None:
    model_assets = _text("docs/MODEL_ASSETS.md")
    optional_commands = (
        "bash get_weights.sh",
        "gdown --folder",
        "hf download Wan-AI/Wan2.2-TI2V-5B",
    )
    for command in optional_commands:
        assert command in model_assets
        assert command not in _text("README.md")
        assert command not in _text("INSTALL.md")
        assert command not in _text("docs/BENCHMARK.md")


def test_configuration_has_one_detailed_owner() -> None:
    configuration = _text("docs/CONFIGURATION.md")
    config_index = _text("configs/README.md")

    for filename in (
        "workspace.json",
        "run.json",
        "credentials.local.json",
        "models.json",
        "runtime.json",
    ):
        assert filename in configuration
    assert "### Root keys" in configuration
    assert "### Root keys" not in config_index
    assert "](../docs/CONFIGURATION.md" in config_index


def test_depth_model_status_is_not_overclaimed() -> None:
    assets = _text("docs/MODEL_ASSETS.md")
    assert "`dvd_official`" in assets
    assert "`dvd_base` is an alias" in assets
    assert "`dvd_lora_specific`" in assets
    assert "`dvd_lora_shared`" in assets
    assert "`vda_metric` / `vda_non_metric`" in assets
    assert "Plain `dvd` remains the low-level backend" in assets


def test_public_docs_use_repository_local_default_data() -> None:
    for relative in (
        "README.md",
        "INSTALL.md",
        "configs/README.md",
        "docs/BENCHMARK.md",
        "docs/CONFIGURATION.md",
    ):
        text = _text(relative)
        assert "../data/dream-exe" not in text, relative


def test_evaluation_guide_exposes_single_case_vlm_evaluation() -> None:
    evaluation = _text("docs/EVALUATION.md")

    assert "## VLM judges" in evaluation
    assert "## Paper VLM judges" not in evaluation
    assert "### Evaluate one case" in evaluation
    assert "### Evaluate the complete benchmark" in evaluation
    assert "dream-exe evaluate visual" in evaluation
    assert "--scope one --case rc_cheesybread_ep000001" in evaluation
    assert "--scope all" in evaluation
