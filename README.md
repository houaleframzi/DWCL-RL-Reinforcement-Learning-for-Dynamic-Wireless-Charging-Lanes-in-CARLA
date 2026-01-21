# DWCL-RL: Reinforcement Learning for Dynamic Wireless Charging Lanes in CARLA

## Overview

This repository contains the **official research code** accompanying the paper:

> **Optimal Control of Electric Vehicles in Dynamic Wireless Charging Lanes Using Deep Q-Networks**  
> Ahmed-Ramzi Houalef, Florian Delavernhe, Sidi-Mohamed Senouci, El-Hassane Aglzim  
> *IEEE 102nd Vehicular Technology Conference (VTC2025-Fall), 2025*

This work addresses **energy-aware optimal control of electric vehicles (EVs)** operating on  
**Dynamic Wireless Charging Lanes (DWCL)** using **Deep Reinforcement Learning (DQN)** in the  
**CARLA high-fidelity simulator**.

The objective is to learn *when* an EV should:
- enter or exit a DWCL,
- adapt its speed while charging,

to satisfy **energy constraints**, **travel-time objectives**, and **lane-switching stability**,  
without relying on handcrafted rules.

---

## 🎥 Video Demonstration

A full demonstration of the trained agent operating in CARLA (DWCL entry/exit manoeuvres, charging
behavior, and energy-aware decisions) is available here:

<p align="center">
  <a href="https://YOUR_VIDEO_LINK_HERE">
    <img src="figures/video_preview.png" width="800">
  </a>
</p>

<p align="center"><em>Click the image to watch the video demonstration.</em></p>

> **Note:** GitHub does not support embedded video playback in Markdown.  
> The video is accessed via a clickable preview image (standard GitHub practice).

---

## Concept and System Overview

### Dynamic Wireless Charging Lane (DWCL)

Dynamic Wireless Charging Lanes allow EVs to recharge **while driving**, eliminating the need for
static charging stops.

<p align="center">
  <img src="figures/dwcl_concept.png" width="700">
</p>
<p align="center"><em>Figure 1: Concept of Dynamic Wireless Charging Lane (DWCL).</em></p>

---

### Control Architecture

<p align="center">
  <img src="figures/system_architecture.png" width="800">
</p>
<p align="center"><em>Figure 2: Overall system architecture combining CARLA, DWCL, battery model, and DQN agent.</em></p>

---

## Reinforcement Learning Formulation

### State Space


- `SoC`: current battery state of charge  
- `SoC_required`: minimum SoC needed to reach destination  
- `ETA`: estimated time of arrival  
- `remaining_distance`: Euclidean distance to destination  
- `lane_type`: 0 = normal lane, 1 = DWCL  
- `target_speed`: current cruising speed (km/h)

---

### Action Space

| ID | Action |
|----|--------|
| 0  | Enter DWCL |
| 1  | Exit DWCL |
| 2  | Accelerate inside DWCL |
| 3  | Decelerate inside DWCL |
| 4  | Maintain speed inside DWCL |
| 5  | Stay outside DWCL (Traffic Manager autopilot) |

**Important:**  
DWCL entry and exit actions are executed as **blocking manoeuvres** using a CARLA `BehaviorAgent`.  
During these manoeuvres, **the DQN agent does not predict new actions**.

---

## Reward Function Design

<p align="center">
  <img src="figures/reward_structure.png" width="750">
</p>
<p align="center"><em>Figure 3: Reward components used for training the DQN agent.</em></p>

The reward balances:
- charging efficiency,
- SoC safety constraints,
- speed limits inside DWCL,
- travel time relative to ETA,
- lane-switching stability,
- terminal success or failure.

The implementation is **identical to the reward formulation described in the paper**, ensuring
full reproducibility.

---

## CARLA Simulation Environment

<p align="center">
  <img src="figures/carla_scene.png" width="800">
</p>
<p align="center"><em>Figure 4: CARLA simulation environment with Dynamic Wireless Charging Lane.</em></p>

The environment implements:
- realistic vehicle dynamics,
- DWCL coil placement,
- Traffic Manager–controlled cruising outside DWCL,
- notebook-faithful manoeuvre execution.

---

## Training Pipeline

<p align="center">
  <img src="figures/training_pipeline.png" width="800">
</p>
<p align="center"><em>Figure 5: DQN training pipeline with experience replay and target network.</em></p>

Key features:
- Per-episode epsilon decay
- Experience replay buffer
- Target network stabilization
- Fully resumable checkpoints (episode, epsilon, optimizer, networks)

---

---

## Citation
If you use this code or build upon this work, please cite the following paper:

```bash

@inproceedings{houalef2025dwcl,
  title     = {Optimal Control of Electric Vehicles in Dynamic Wireless Charging Lanes Using Deep Q-Networks},
  author    = {Houalef, Ahmed-Ramzi and Delavernhe, Florian and Senouci, Sidi-Mohamed and Aglzim, El-Hassane},
  booktitle = {IEEE 102nd Vehicular Technology Conference (VTC2025-Fall)},
  year      = {2025},
  address   = {Chengdu, China},
  publisher = {IEEE},
  doi       = {10.1109/VTC2025-Fall65116.2025.11310467},
  hal_id    = {hal-05094364}
}


Preprint available at:
https://hal.science/hal-05094364v1
'''



