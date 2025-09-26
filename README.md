# Enter the Mind Palace: Reasoning and Planning for Long-term Active Embodied Question Answering

[[paper](https://www.arxiv.org/pdf/2507.12846)]
[[project](https://mind-palace-laeqa.github.io/)]


## Abstract
As robots become increasingly capable of operating over extended periods—spanning days, weeks, and even months—they are expected to accumulate knowledge of their environments and leverage this experience to assist humans more effectively. This paper studies the problem of Long-term Active Embodied Question Answering (LA-EQA), a new task in which a robot must both recall past experiences and actively explore its environment to answer complex, temporally-grounded questions. Unlike traditional EQA settings, which typically focus either on understanding the present environment alone or on recalling a single past observation, LA-EQA challenges an agent to reason over past, present, and possible future states, deciding when to explore, when to consult its memory, and when to stop gathering observations and provide a final answer. Standard EQA approaches based on large models struggle in this setting due to limited context windows, absence of persistent memory, and an inability to combine memory recall with active exploration.

To address this, we propose a structured memory system for robots, inspired by the mind palace method from cognitive science. Our method encodes episodic experiences as scene-graph-based world instances, forming a reasoning and planning algorithm that enables targeted memory retrieval and guided navigation. To balance the exploration-recall trade-off, we introduce value-of-information-based stopping criteria that determine when the agent has gathered sufficient information. We evaluate our method on real-world experiments and introduce a new benchmark that spans popular simulation environments and actual industrial sites. Our approach significantly outperforms state-of-the-art baselines, yielding substantial gains in both answer accuracy and exploration efficiency.

## Long-term Active EQA Benchmark
We generate the Long-term Active EQA Benchmark to test agents' understanding of the environment and the changes of the environment in long-term setting across multiple days and months.

We release the LA-EQA question list and the scene files:
1. List of questions: eqa_questions.json
2. Scene files: link (6 GB)




## Citing Mind Palace Exploration and LA-EQA Benchmark

```tex
@inproceedings{ginting2025laeqa,
  author={ Muhammad Fadhil Ginting, Dong-Ki Kim, Xiangyun Meng, Andrzej Reinke, Bandi Jai Krishna,
Navid Kayhani, Oriana Peltzer, David D. Fan, Amirreza Shaban, Sung-Kyun Kim,
Mykel J. Kochenderfer, Ali-akbar Agha-Mohammadi, Shayegan Omidshafiei},
  title={{Enter the Mind Palace: Reasoning and Planning for Long-term Active Embodied Question Answering}},
  booktitle={{Conference on Robot Learning (CoRL)}},
  year={2025},
}
```
