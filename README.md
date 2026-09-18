<p align="center">
  <a href="assets/dream-exe-teaser.png">
    <img src="assets/dream-exe-teaser.gif" width="100%" alt="Dream.exe: Can Video Generation Models Dream Executable Robot Manipulation? — 18 synchronized ground-truth benchmark demonstrations">
  </a>
</p>

<div align="center">

# Dream.exe: Can Video Generation Models Dream Executable Robot Manipulation?

**Rui Zhao**<sup>1,\*</sup>, **Kaiming Yang**<sup>1,\*</sup>, **Jifeng Zhu**<sup>1,†</sup>, **Siyang Chen**<sup>1,†</sup>, **Ziqi Wang**<sup>1</sup>, **Weijia Wu**<sup>1</sup>, **Kevin Qinghong Lin**<sup>2</sup>, **Heng Wang**<sup>3</sup>, **Mike Zheng Shou**<sup>1,‡</sup>

<sup>1</sup>Show Lab, National University of Singapore &nbsp;&nbsp; <sup>2</sup>University of Oxford &nbsp;&nbsp; <sup>3</sup>Tencent

<sub><sup>\*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Equal contribution (second authors) &nbsp;&nbsp; <sup>‡</sup>Corresponding author</sub>

</div>

---

> [!IMPORTANT]
> **🚧 Code, benchmark data, and evaluation tools will be open-sourced here. Stay tuned!**
> Please ⭐ **star** and **watch** this repository to be notified when the release lands.

---

## 📖 Overview

> **Can a video generation model's dream of manipulation actually be *executed* by a robot?**

Dream.exe answers this by taking generated videos out of the screen and into a physics simulator. Instead of judging a video only by how good it looks, we convert the motion it depicts into a robot trajectory, execute it, and measure whether the task actually succeeds. Execution success then becomes a grounding signal that purely visual metrics cannot offer.

**What's inside:**

- 🎬 **Video-to-execution pipeline.** From a single scene image and task prompt, we generate a manipulation video, lift it into a 3D robot trajectory, and roll it out in simulation.
- 🧪 **101-task benchmark.** Manually curated from RoboCasa and stratified into three levels of physical complexity, scored on visual quality, trajectory fidelity, and execution success.
- 🤖 **8 models evaluated.** Frontier closed-source, open-source, and robot-specific video generators under one unified protocol.

**Key findings:**

- ✅ Generative priors from internet-scale data already encode meaningful physical knowledge. Several models achieve measurable execution success with no robot-specific supervision.
- ⚠️ Visual quality is a poor predictor of executability. Physical-plausibility scores barely correlate with task success.
- 🧗 Long-horizon tasks remain hard. Multi-stage manipulation exposes the limits of current models.

## 🧪 Benchmark Task Suite

<div align="center">
  <img src="assets/task_suite.png" width="100%" alt="Overview of the Dream.exe task suite">
</div>

**Overview of the Dream.exe task suite.** *Left:* representative scenes and task prompts from each difficulty level. *Top right:* distribution of 101 tasks across the three levels. *Bottom right:* camera viewpoints are deliberately diversified across scenes to improve generalization coverage.

The tasks are stratified into three levels of increasing physical complexity:

- **Level 1, Single-object manipulation.** Geometrically consistent end-effector motion with correct grasp/release timing.
- **Level 2, Multi-object interaction.** Reasoning about object-to-object relationships and placement.
- **Level 3, Multi-stage composite tasks.** Maintaining physical coherence across a long task horizon with correctly sequenced sub-goals.

## 🦾 Video-to-Trajectory Pipeline

<div align="center">
  <img src="assets/video2traj_pipeline.png" width="100%" alt="Dream.exe pipeline: video generation, depth estimation, 2D tracking, 3D trajectory recovery, and robot execution">
</div>

- 🎬 **Generate & track.** Start from a scene image and task prompt, then track the end-effector and manipulated object throughout the generated video.
- 📐 **Lift into 3D.** Our LoRA-adapted DVD estimates depth, calibrated with the initial scene depth; camera parameters lift tracked pixels into world coordinates.
- 🦾 **Recover actions.** Calibrate the end-effector trajectory to the robot's control point, estimate rotation, and infer grasp/release timing from end-effector–object motion.
- ✅ **Execute & evaluate.** Combine position, rotation, and gripper state into a 7D trajectory, then execute it with closed-loop control in simulation to measure task success.

See [Section 3.3 and Appendix C of our paper](https://arxiv.org/html/2606.04811v1#S3.SS3) for details.

## 📌 Citation

If you find our work useful, please consider citing:

```bibtex
@article{zhao2026dreamexe,
  title   = {Dream.exe: Can Video Generation Models Dream Executable Robot Manipulation?},
  author  = {Zhao, Rui and Yang, Kaiming and Zhu, Jifeng and Chen, Siyang and Wang, Ziqi and Wu, Weijia and Lin, Kevin Qinghong and Wang, Heng and Shou, Mike Zheng},
  year    = {2026}
}
```
