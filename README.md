# Verifier-Guided Post-Training for Arithmetic Reasoning

A proposed study of whether supervised examples, preference learning, and reinforcement learning improve a language model's ability to solve arithmetic problems under exact constraints.

**Repository status:** project design and evaluation specification. This repository currently contains documentation; training code, datasets, checkpoints, and measured results are not included.

## The task

Given four or five numbers and a target, the model must produce an expression using every supplied number exactly once. Only addition, subtraction, multiplication, division, and parentheses are permitted. For example, the numbers `2, 3, 4, 5` can reach `24` through `(5 + 3 - 2) * 4`.

A plausible expression can still be invalid: it may repeat a number, introduce an extra constant, divide by zero, or miss the target. This makes arithmetic a useful setting for separating fluent output from verifiable correctness. The intended scope is constrained expression generation; conclusions would apply to this task rather than general mathematical reasoning.

## Proposed approach

The study is scoped around **Qwen2.5-7B-Instruct**, with PyTorch, Transformers, TRL, and PEFT/LoRA as the intended training stack.

| Stage | Planned data | Purpose |
|---|---|---|
| Supervised fine-tuning (SFT) | 12,000 solver-verified demonstrations | Learn the output format and valid solution patterns |
| Direct preference optimization (DPO) | 6,000 pairs from same-problem samples | Prefer valid solutions over incorrect or constraint-violating answers |
| Group relative policy optimization (GRPO) | Groups of newly sampled solutions | Update the policy using exact, programmatic rewards |

The verifier would parse expressions with Python's abstract syntax tree and evaluate them with `fractions.Fraction`. It would accept only permitted syntax, compare the multiset of integer leaves with the input numbers, reject division by zero, and test exact equality with the target. A reward of one would require every check to pass; all other answers would receive zero. Unverified prose explanations would not count as evidence of correct reasoning.

For DPO, chosen and rejected answers would come from the same problem, with the SFT checkpoint serving as the reference policy. For GRPO, the study would track groups containing only successes or only failures, since those groups provide little within-group reward contrast.

## Evaluation design

Input numbers would be grouped by their sorted multiset before splitting. Different permutations or targets derived from one number combination would remain in the same split. Four-number and five-number tasks would both appear in the evaluation, with separate difficulty summaries.

The planned final set contains 1,000 held-out tasks. Every checkpoint would use the same prompt, greedy single-attempt decoding, and output-length limit. The verifier would score outputs without supplying corrections or selecting a successful answer from multiple attempts.

Comparisons would cover the base model, SFT, SFT→DPO, SFT→GRPO, and SFT→DPO→GRPO. Strict success would require both exact arithmetic and valid number use. A matched-training-budget comparison is necessary before attributing any advantage specifically to DPO. No measured success rates are reported in this repository.

## References and provenance

- [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer)

This design and its documentation were prepared with AI assistance. The repository presents a proposed implementation and does not establish completed personal training or experimental contributions.
