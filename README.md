# AI Canteen Optimization System

## Overview

The AI Canteen Optimization System is a reinforcement learning-based application designed to optimize queue management and service counter allocation in a dynamic canteen environment.

The system uses a Q-Learning agent to make real-time decisions that minimize queue length while maintaining efficient use of available counters.

---

## Objective

- Reduce waiting time for customers  
- Optimize the number of active service counters  
- Maintain a balanced and efficient system under varying demand conditions  

---

## Problem Statement

In environments such as college or office canteens, demand fluctuates throughout the day. Static allocation of service counters leads to inefficiencies such as:

- Long queues during peak hours  
- Underutilized counters during low demand  

This project addresses the problem by introducing an adaptive, learning-based system that dynamically adjusts decisions based on real-time conditions.

---

## Solution Approach

The system models the canteen as a reinforcement learning environment and applies Q-Learning to determine optimal actions.

### State Representation
- Queue length (bucketed into ranges)
- Number of active counters
- Time period (low, medium, peak)

### Action Space
- No Action  
- Open Counter  
- Speed Service  
- Close Counter  

### Learning Mechanism
- Q-Learning algorithm  
- Epsilon-greedy exploration strategy  
- Reward-based learning  

The agent continuously updates its policy to improve long-term performance.

---

## System Architecture

- `environment.py`  
  Defines the simulation environment, queue dynamics, and reward function.

- `agent.py`  
  Implements the Q-Learning algorithm, action selection, and learning updates.

- `service.py`  
  Handles simulation execution and result aggregation.

- `app.py`  
  Provides a Gradio-based user interface for interactive simulation.

- `main.py`  
  Exposes FastAPI endpoints for programmatic access.

- `train.py`  
  Handles training of the reinforcement learning agent.

- `model_loader.py`  
  Manages saving and loading of the trained Q-table.

- `utils.py`  
  Contains helper functions such as efficiency calculation.

- `config.py`  
  Stores configurable parameters such as learning rates and thresholds.

- `visualize.py`  
  Generates plots for training and simulation analysis.

---

## Features

- Dynamic counter management based on demand  
- Reinforcement learning-driven decision system  
- Real-time simulation interface  
- Efficiency scoring and performance tracking  
- Queue trend visualization  
- Modular and extensible design  

---

## Tech Stack

- Python  
- Gradio (User Interface)  
- FastAPI (Backend API)  
- Matplotlib (Visualization)  
- Q-Learning (Reinforcement Learning)  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Saravanan-246/canteen-rl.git
cd canteen-rl