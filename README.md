# DWCL-RL: Reinforcement Learning for Dynamic Wireless Charging Lanes in CARLA

## Overview

This repository contains the **official research code** accompanying the paper:

> **Optimal Control of Electric Vehicles in Dynamic Wireless Charging Lanes Using Deep Q-Networks**  
> Ahmed-Ramzi Houalef, Florian Delavernhe, Sidi-Mohamed Senouci, El-Hassane Aglzim  
> IEEE 102nd Vehicular Technology Conference (VTC2025-Fall), 2025

This work studies **energy-aware optimal control of electric vehicles (EVs)** operating on **Dynamic Wireless Charging Lanes (DWCL)** using **Deep Reinforcement Learning (DQN)** in the **CARLA simulator**.

The goal is to learn *when* an EV should:
- enter or exit a DWCL,
- adapt its speed while charging,

in order to satisfy **energy constraints**, **travel-time objectives**, and **lane-switching stability**, without relying on handcrafted rules.

---

## Concept and System Overview

### Dynamic Wireless Charging Lane (DWCL)

Dynamic Wireless Charging Lanes allow EVs to recharge **while driving**, removing the need for static charging stops.

<p align="center">
  <img src="figures/dwcl_concept.png" width="700">
</p>
<p align="center"><em>Figure 1: Concept of Dynamic Wireless Charging Lane (DWCL).</em></p>

---

### Control Architecture

The proposed system combines:
- a **CARLA-based high-fidelity simulator**,
- a **learned battery power consumption model**,
- a **DQN agent** making discrete control decisions.

<p align="center">
  <img src="figures/system_architecture.png" width="800">
</p>
<p align="center"><em>Figure 2: Overall system architecture combining CARLA, DWCL, battery model, and DQN agent.</em></p>

---

## Reinforcement Learning Formulation

### State Space

The agent observes the following state vector:

