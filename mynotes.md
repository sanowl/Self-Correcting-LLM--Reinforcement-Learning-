
# SCoRe System Technical Notes

## Core Components

### 1. Model Architecture
- Uses LLaMA (Large Language Model Meta AI) as the base model
- Wrapped in an `AdvancedModel` class for easier handling

### 2. Training Process
- Two-stage approach:
  1. Policy Initialization (Stage I)
  2. Multi-Turn Reinforcement Learning (Stage II)

### 3. Reward Mechanism
- Task-specific reward functions:
  - Math: Symbolic equation checking
  - Code: Safe execution and test case validation

### 4. KL Divergence Penalty
- Maintains similarity to reference model
- Prevents drastic departures from initial policy

## Key Algorithms

### Stage I: Policy Initialization
1. Generate initial attempts
2. Compute rewards
3. Apply KL divergence penalty
4. Update model parameters

### Stage II: Multi-Turn RL
1. Generate first attempt
2. Create prompt for correction
3. Generate second attempt
4. Compute rewards for both attempts
5. Apply reward shaping
6. Compute KL divergence
7. Update model parameters

### Reward Shaping
- Bonus for improvement: `α * (reward_second - reward_first)`
- Encourages self-correction behavior

### Safe Code Execution
- Runs generated code in a separate thread
- Implements timeout mechanism for safety

## Training Details

- Uses AdamW optimizer
- Linear learning rate schedule with warmup
- Gradient accumulation for effective larger batch sizes
- Mixed precision training option for efficiency

## Evaluation Metrics

- Accuracy@t1: First attempt accuracy
- Accuracy@t2: Second attempt accuracy
- Δ(t1,t2): Overall improvement
- Δ_i→c(t1,t2): Incorrect to correct ratio
- Δ_c→i(t1,t2): Correct to incorrect ratio

## Data Handling

- Custom datasets for MATH, MBPP, and HumanEval tasks
- Dynamic batch preparation based on task type

## Visualization

- Training reward history plotting
- Edit distance ratio visualization between attempts