from __future__ import annotations

import subprocess
from pathlib import Path

from dream_exe.bench.data.workspace import Workspace
from dream_exe.bench.videos import generation as generation_module
from dream_exe.bench.videos import import_video as import_module
from dream_exe.bench.videos.preprocess import preprocess_video_for_video2traj
from dream_exe.generation.video import BaseImageToVideoBackend


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART_BENCH = REPOSITORY_ROOT / "examples/quickstart/data/bench"


def _workspace(tmp_path: Path) -> Workspace:
    return Workspace(
        config_path=tmp_path / "workspace.json",
        bench_root=QUICKSTART_BENCH,
        published_results_root=tmp_path / "published",
        outputs_root=tmp_path / "outputs",
        work_root=tmp_path / "work",
        archive_root=tmp_path / "archive",
        external_root=tmp_path / "external",
        checkpoint_root=tmp_path / "checkpoints",
        bindings={},
    )


def _make_video(path: Path, *, size: str = "64x32") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s={size}:r=5:d=0.6",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            path.as_posix(),
        ),
        check=True,
    )


class _VideoBackend(BaseImageToVideoBackend):
    backend_id = "test-video-backend"

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
        output_path: Path,
        seed: int,
        parameters: dict,
    ) -> dict:
        del image_path, prompt, seed, parameters
        _make_video(output_path)
        return {"revision": "test"}


def test_shared_preprocessor_creates_validated_square_derivative(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "preprocessed.mp4"
    _make_video(source)

    result = preprocess_video_for_video2traj(
        source,
        destination,
        output_contract={"pipeline_width": 32, "pipeline_height": 32},
    )

    assert result["status"] == "processed"
    assert result["spec"]["mode"] == "resize"
    assert result["media"]["width"] == 32
    assert result["media"]["height"] == 32
    assert result["media"]["codec"] == "h264"
    assert result["media"]["pixel_format"] == "yuv420p"
    assert source.is_file()


def test_import_always_registers_raw_and_preprocessed_videos(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = _workspace(tmp_path)
    source = tmp_path / "my-video.mp4"
    _make_video(source)
    captured = {}

    def register(**kwargs):
        captured.update(kwargs)
        return {"status": "registered-for-test"}

    monkeypatch.setattr(import_module, "register_video_output", register)
    result = import_module.import_external_video(
        workspace=workspace,
        uid="rc_cheesybread_ep000001",
        model_id="MyVideoModel",
        prompt_variant="standard",
        video_path=source,
    )

    assert result["status"] == "registered-for-test"
    assert result["source_preserved"] is True
    assert source.is_file()
    assert set(captured["source_paths"]) == {"video", "preprocessed"}
    assert captured["manifest"]["preprocessed"]["path"] == "preprocessed.mp4"
    assert captured["manifest"]["origin"]["preprocessing_kind"] == "automatic"
    assert captured["manifest"]["origin"]["preprocessing"]["width"] == 512

def test_generation_always_registers_raw_and_preprocessed_videos(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = _workspace(tmp_path)
    captured = {}

    def register(**kwargs):
        captured.update(kwargs)
        return {"status": "registered-for-test"}

    monkeypatch.setattr(generation_module, "register_video_output", register)
    result = generation_module.generate_video_output(
        workspace=workspace,
        uid="rc_cheesybread_ep000001",
        model_id="Wan2.2",
        prompt_variant="standard",
        backend=_VideoBackend(),
        backend_name="Wan2.2",
    )

    assert result["status"] == "registered-for-test"
    assert set(captured["source_paths"]) == {"video", "preprocessed"}
    assert captured["manifest"]["preprocessed"]["path"] == "preprocessed.mp4"
    assert captured["manifest"]["origin"]["preprocessing"]["width"] == 512
