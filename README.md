# GFN-SR: Symbolic Regression with Generative Flow Networks

by
Sida Li,
Ioana Levinscu,
Sebastian Musslick

Accepted at [NeurIPS 2023 AI4Science Workshop](https://ai4sciencecommunity.github.io/neurips23.html)

> Source code released soon due to main authors being busy earlier. Stay tuned~
> 
> This page currently only contains future roadmap for this project. 

## Future Project Roadmap
- [ ] Redesign GFlowNets state space for SR
- [ ] Improve LSTM forward policy network
- [ ] Find alternative reward function / reward baseline
- [ ] Benchmark other training losses for GFlowNets (e.g. detailed balance, flow matching)

## GFlowNet Backgrounds
- [Detailed paper from MILA](https://arxiv.org/abs/2111.09266)
- [Tutorial post on GFN](https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b3)
- [Other awesome GFlowNet projects](https://github.com/zdhNarsil/Awesome-GFlowNets)

## Abstract

> Symbolic regression (SR) is an area of interpretable machine learning that aims to identify mathematical expressions, often composed of simple functions, that best fit in a given set of covariates $X$ and response $y$. In recent years, deep symbolic regression (DSR) has emerged as a popular method in the field by leveraging deep reinforcement learning to solve the complicated combinatorial search problem. In this work, we propose an alternative framework (GFN-SR) to approach SR with deep learning. We model the construction of an expression tree as traversing through a directed acyclic graph (DAG) so that GFlowNet can learn a stochastic policy to generate such trees sequentially. Enhanced with an adaptive reward baseline, our method is capable of generating a diverse set of best-fitting expressions. Notably, we observe that GFN-SR outperforms other SR algorithms in noisy data regimes, owing to its ability to learn a distribution of rewards over a space of candidate solutions.

## Cite this work
```
@misc{li2023gfnsr,
      title={GFN-SR: Symbolic Regression with Generative Flow Networks}, 
      author={Sida Li and Ioana Marinescu and Sebastian Musslick},
      year={2023},
      eprint={2312.00396},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
