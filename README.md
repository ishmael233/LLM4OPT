# LLM4OPT: from Modeling to Solving

## 🎯 LLMs for Optimization

Optimization techniques have become a core tool for solving complex problems across diverse fields such as engineering design, economic planning, and scientific discovery. When tackling real-world challenges, researchers typically abstract them into mathematical optimization models, which are then solved using various optimization algorithms. In recent years, large language models (LLMs) have emerged as a powerful paradigm, offering novel perspectives and capabilities for the field of optimization. The application of LLMs in optimization spans two primary areas: **modeling** and **solving**.

We provide a **classification** and **collection** of these applications to serve as a reference for researchers. We propose a new taxonomy to clearly articulate the deep relationship between LLMs and the optimization process. Specifically, we classify the research in this field into two main categories: LLMs for optimization modeling and LLMs for optimization solving. The latter, LLMs for optimization solving, is further subdivided into three classes: LLMs as optimizers, low-level LLMs assisted optimization algorithms, and high-level LLMs assisted optimization algorithms. The specific definitions for these are as follows:


1. 💻 **LLMs for Optimization Modeling** aims to automatically transform unstructured natural language descriptions of problems into mathematical optimization models that can be understood and solved by machines. This research area is the foundational step toward achieving a fully LLM-driven optimization workflow, with the core challenge lying in bridging the gap between ambiguous natural language and precise mathematical models. Two main paradigms exist in this field: prompt-based and learning-based methods. The former typically relies on carefully designed prompting procedures or the collaboration of multiple agents, with the advantage of being quickly deployable without additional training. The latter, however, often requires data synthesis and model fine-tuning. While these methods incur training costs, they can significantly enhance modeling reliability in complex scenarios.

2. 🤖 **LLMs as Optimizers** refers to the use of LLMs directly as general-purpose optimizers, which solve optimization problems through iterative natural language interaction without relying on traditional optimization algorithm frameworks. This paradigm leverages the powerful in-context learning and complex reasoning capabilities of LLMs to analyze optimization trajectories and continuously generate more promising candidate solutions. Although this direct optimization approach faces challenges such as model dependency, it stands as the earliest bridge connecting large models to optimization tasks, possessing significant pioneering importance and potential.

3. 🔧 **Low-level LLMs for Optimization Algorithms** refers to the paradigm of embedding LLMs as intelligent components within traditional optimization algorithms to enhance their specific operations. Unlike acting as a standalone optimizer, this approach tightly integrates LLMs with traditional evolutionary algorithms, leveraging the models' powerful data analysis capabilities and rich domain knowledge to boost algorithmic performance. Specifically, LLMs can be applied to core processes such as algorithm initialization, evolutionary operators, algorithm configuration, and fitness evaluation. This method relies on an efficient optimization framework to empower traditional components, enabling them to adapt more intelligently to different problem characteristics and search states, thereby improving the algorithm's efficiency and solution quality at a fundamental level.

4. 🎨 **High-level LLMs for Optimization Algorithms** differs from the low-level assistance paradigm by focusing on the top-level orchestration or design of algorithms, rather than their internal components. Specifically, high-level assistance involves two types of tasks: algorithm selection and algorithm generation. The algorithm selection task aims to match the most suitable algorithm from a pool for different problem instances, whereas algorithm generation goes a step further, requiring LLMs to autonomously design new algorithms to better fit a given problem. This class of methods grants LLMs a global perspective over the entire optimization task, transforming their role from an internal assistant to a top-level designer for the entire optimization workflow.

## 🤝 How to Contribute

We welcome any suggestions and pull requests! This list is far from comprehensive, and we encourage the community to help us improve it.

### 📋 Ways to Contribute

1. 🛠️ **Fork, Add, and Merge** 
   - Fork this repository
   - Add your suggestions or updates
   - Submit a pull request for review

2. 🐛 **Report Issues** 
   - Found a problem or have a suggestion?
   - Open an issue to let us know

3. 📧 **Direct Contact** 
   - Have questions or want to collaborate?
   - Contact Yisong Zhang at [23s004040@stu.hit.edu.cn](mailto:23s004040@stu.hit.edu.cn)

### 📜 Sharing Principle

The references listed here are shared for **research purposes only**. If you are an author and do not wish your work to be listed here, please feel free to contact us. We will promptly remove the content upon request.

---

⭐ **We appreciate your contributions to making this resource better for everyone!**


## 📑 Table of Contents

- 📚 [1. Survey Papers](#1-survey-papers)
- 💻 [2. LLMs for Optimization Modeling](#2-LLMs-for-Optimization-Modeling)
  - 🎮 [2.1 Prompt-based Methods](#21-Prompt-based-Methods)
  - 🧠 [2.2 Learning-based Methods](#22-Learning-based-Methods)
- ⚙️ [3. LLMs for Optimization Solving](#3-LLMs-for-Optimization-Solving)
  - 🤖 [3.1 LLMs as Optimizers](#31-LLMs-as-Optimizers)
  - 🔧 [3.2 Low-level LLMs for Optimization Algorithms](#32-Low-level-LLMs-for-Optimization-Algorithms)
  - 🎨 [3.3 High-level LLMs for Optimization Algorithms](#33-High-level-LLMs-for-Optimization-Algorithms)


## 1. Survey Papers
| **Title** | **Publication** | **Year** |
|-----------|--------------|------|
| [Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap](https://arxiv.org/abs/2401.10034) | TEVC | 2024 |
| [When Large Language Model Meets Optimization](https://arxiv.org/abs/2405.10098) | SWEVO | 2024 |
| [When Large Language Models Meet Evolutionary Algorithms](https://arxiv.org/abs/2401.10510) | Research | 2024 |
| [Exploring the Improvement of Evolutionary Computation via Large Language Models](https://arxiv.org/abs/2405.02876) | GECCO | 2024 |
| [A Systematic Survey on Large Language Models for Algorithm Design](https://arxiv.org/abs/2410.14716) | arXiv | 2024 |
| [Deep Insights into Automated Optimization with Large Language Models and Evolutionary Algorithms](https://arxiv.org/abs/2410.20848) | arXiv | 2024 |
| [Toward Automated Algorithm Design: A Survey and Practical Guide to Meta-Black-Box-Optimization](https://arxiv.org/abs/2411.00625) | TEVC | 2024 |
| [Evolutionary Computation and Large Language Models: A Survey of Methods, Synergies, and Applications](https://arxiv.org/abs/2505.15741) | arXiv | 2025|
| [A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions](https://arxiv.org/abs/2508.10047) | arXiv | 2025 |


## 2. LLMs for Optimization Modeling
We'll delve into two core paradigms for using LLMs for optmization modeling: prompt-based methods and learning-based methods. Prompt-based methods leverage carefully designed instructions to guide the LLM in completing modeling tasks. This often involves a two-stage workflow or the collaboration of multiple LLMs. Learning-based methods, on the other hand, use fine-tuned models to directly generate mathematical models. However, they typically require pre-training on a large amount of synthetic data.

<div style="text-align: center;">
    <img src="fig/formulation.png" width="600" />
</div>

### 2.1 Prompt-based Methods

| **Title** | **Publication** | **Year** | **Type** | **Summary** |
|-----------|--------------|------|------|----------|
| [NER4OPT: Named Entity Recognition for Optimization Modelling from Natural Language](https://link.springer.com/chapter/10.1007/978-3-031-33271-5_20) | CPAIOR | 2023 | Two-stage | Fine-tune models for named entity recognition by incorporating traditional NLP methods. |
| [Towards an Automatic Optimisation Model Generator Assisted with Generative Pre-trained Transformer](https://arxiv.org/abs/2305.05811) | GECCO | 2023 | Direct | Directly leverage LLMs to generate mathematical models. |
| [Holy Grail 2.0: From Natural Language to Constraint Models](https://arxiv.org/abs/2308.01589) | arXiv | 2023 |Two-stage | Embed LLMs within the two-stage framework. |
| [Synthesizing mixed-integer linear programming models from natural language descriptions](https://arxiv.org/abs/2311.15271) | arXiv | 2023 | Two-stage | Employ fine-tuned models for constraint classification. |
| [Chain-of-experts: When llms meet complex operations research problems](https://openreview.net/pdf?id=HobyL1B9CZ) | ICLR | 2023 | Multi-agent | Construct dynamic reasoning chains using 11 expert agents. |
| [OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models](https://arxiv.org/abs/2402.10172) | ICML | 2023 | Multi-agent | Develop a conductor agent to coordinate modeling, programming, and evaluation processes. |
| [Optimization modeling and verification from problem specifications using a multi-agent multi-stage LLM framework](https://www.tandfonline.com/doi/abs/10.1080/03155986.2024.2381306) | INFOR | 2024 | Multi-agent | Develop inter-agent cross-validation to replace solver-dependent verification. |
| [Democratizing energy management with llm-assisted optimization autoformalism](https://ieeexplore.ieee.org/abstract/document/10738100) | SGC | 2024 | Two-stage | Apply the two-stage framework to energy management systems. |
| [CAFA: Coding as Auto-Formulation Can Boost Large Language Models in Solving Linear Programming Problem](https://openreview.net/pdf?id=xC2xtBLmri) | NeurIPS | 2024 | Two-stage | Enhance modeling performance through code-based formalization. |
| [Llms for mathematical modeling: Towards bridging the gap between natural and mathematical languages](https://arxiv.org/abs/2405.13144) | NAACL | 2024 | Two-stage | Develop the MAMO benchmark with ordinary differential equation extensions. |
| [Abstract Operations Research Modeling Using Natural Language Inputs](https://www.mdpi.com/2078-2489/16/2/128) | arXiv | 2024 | Two-stage | Predefine abstract structural constraints to regulate the LLM outputs. |
| [TRIP-PAL: Travel Planning with Guarantees by Combining Large Language Models and Automated Planners](https://arxiv.org/abs/2406.10196) | arXiv | 2024 | Two-stage | Apply the two-stage framework to travel planning. |
| [Solving General Natural-Language-Description Optimization Problems with Large Language Models](https://arxiv.org/abs/2407.07924) | arXiv | 2024 | Interactive | Support both single-input and interactive-input modes. |
| [OptiMUS-0.3: Using Large Language Models to Model and Solve Optimization Problems at Scale](https://arxiv.org/abs/2407.19633) | arXiv | 2024 | Multi-agent | Introduce self-corrective prompts and structure-aware modeling based on OptiMUS. |
| [“I Want It That Way”: Enabling Interactive Decision Support Using Large Language Models and Constraint Programming](https://dl.acm.org/doi/abs/10.1145/3685053) | TiiS | 2024 | Interactive | Develop an interactive system for user input processing with five selectable task options. |
| [Values in the Loop: Designing Interactive Optimization with Conversational Feedback](https://dl.acm.org/doi/abs/10.1145/3719160.3735655) | CUI | 2025 | Interactive | Convert user priorities into optimization constraints during dialog interactions. |
| [Autoformulation of Mathematical Optimization Models Using LLMs](https://arxiv.org/abs/2411.01679) | arXiv | 2025 | Two-stage | Conduct hierarchical Monte Carlo tree search over the hypothesis space. |
| [EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations](https://arxiv.org/abs/2502.14760) | arXiv | 2025 | Two-stage | Generate variable mapping functions via LLMs with lightweight verification. |
| [Large Language Model-Based Automatic Formulation for Stochastic Optimization Models](https://arxiv.org/abs/2508.17200) | arXiv | 2025 | Multi-agent | Deploy multiple independent reviewers to evaluate modeling results. |
| [ORMind: A Cognitive-Inspired End-to-End Reasoning Framework for Operations Research](https://arxiv.org/abs/2506.01326) | arXiv | 2025 | Multi-agent | Replace the conductor agent with structured, predictable workflows. |

### 2.2 learning-based Methods

| **Title** | **Publication** | **Year** | **Type** | **Summary** |
|-----------|--------------|------|------|----------|
| [LM4OPT: Unveiling the Potential of Large Language Models in Formulating Mathematical Optimization Problems](https://arxiv.org/abs/2403.01342) | INFOR | 2024 | Fine-tuning | Progressively fine-tune models on the NL4OPT dataset. |
| [ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling](https://pubsonline.informs.org/doi/abs/10.1287/opre.2024.1233) | OR | 2024 | Data synthesis | Synthesize data through expansion and augmentation and fine-tune open-source models. |
| [OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling](https://arxiv.org/abs/2407.09887) | ICLR | 2024 | Data synthesis | Propose inverse data synthesis methodology and construct the OPTIBENCH benchmark. |
| [LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch](https://arxiv.org/abs/2410.13213) | ICLR | 2024 | Fine-tuning | Introduce model alignment and self-correction mechanisms to mitigate hallucination phenomena. |
| [BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](https://arxiv.org/abs/2411.17404) | arXiv | 2024 | Data synthesis | Solve the problem of missing detailed in data synthesis. |
| [OptMATH: A Scalable Bidirectional Data Synthesis Framework for Optimization Modeling](https://arxiv.org/abs/2502.11102) | arXiv | 2025 | Data synthesis | Develop a scalable bidirectional data synthesis approach. |
| [Language Models for Business Optimisation with a Real World Case Study in Production Schedulingr](https://arxiv.org/abs/2309.13218) | arXiv | 2025 | Data synthesis | Propose a method for fine-tuning cost-effective LLMs to tackle specific business challenges. |
| [Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling](https://arxiv.org/abs/2505.11792) | arXiv | 2025 | Fine-tuning | Integrate external optimization solvers as verifiable reward validators for reinforcement learning. |
| [Step-Opt: Boosting Optimization Modeling in LLMs through Iterative Data Synthesis and Structured Validation](https://arxiv.org/abs/2506.17637) | arXiv | 2025 | Data synthesis | Increase problem complexity through an iterative problem generation approach. |
| [Auto-Formulating Dynamic Programming Problems with Large Language Models](https://arxiv.org/abs/2507.11737) | arXiv | 2025 | Data synthesis | Combine the diversity of forward generation and the reliability of inverse generation. |
| [Toward a Trustworthy Optimization Modeling Agent via Verifiable Synthetic Data Generation](https://arxiv.org/abs/2508.03117) | arXiv | 2025 | Data synthesis | Develop a verifiable synthetic data generation pipeline. |

## 3. LLMs for Optimization Solving

The advent of large language models has propelled this research into a new phase. This section will delve into three core paradigms for utilizing LLMs in optimization solving: LLMs as optimizers, low-level LLMs for optimization algorithms, and high-level LLMs for optimization algorithms.

### 3.1 LLMs as Optimizers

LLMs as optimizers refers to the use of LLMs directly as general-purpose optimizers, which solve optimization problems through iterative natural language interaction without relying on traditional optimization algorithm frameworks. This paradigm leverages the powerful in-context learning and complex reasoning capabilities of LLMs to analyze optimization trajectories and continuously generate more promising candidate solutions.

<div style="text-align: center;">
    <img src="fig/direct.png" width="200" />
</div>


| **Title** | **Publication** | **Year** | **Type** | **Summary** |
|-----------|--------------|------|------|----------|
| [Large Language Models as Optimizers](https://doi.org/10.48550/arXiv.2309.03409) | ICLR | 2023 | Prompt-based | Iteratively refine through optimization trajectories and problem descriptions. |
| [Towards Optimizing with Large Language Models](https://arxiv.org/abs/2310.05204) | KDD | 2023 | Prompt-based | Design four canonical tasks to evaluate the performance boundaries of LLMs. |
| [Large Language Models As Evolution Strategies](https://dl.acm.org/doi/abs/10.1145/3638530.3654238) | GECCO | 2024 | Prompt-based | Replace traditional optimization trajectories with candidate solution quality ranking. |
| [How Multimodal Integration Boost the Performance of LLM for Optimization: Case Study on Capacitated Vehicle Routing Problems](https://doi.org/10.48550/arXiv.2403.01757) | MCII | 2024 | Prompt-based | Utilize MLLMs to jointly process problem descriptions and map visualizations for CVRPs. |
| [Pretrained Optimization Model for Zero-Shot Black Box Optimization](https://proceedings.neurips.cc/paper_files/paper/2024/file/19e9a88d91917775b34fdad447ed8908-Paper-Conference.pdf) | NeurIPS | 2024 | Learning-based | Pre-train a general-purpose, zero-shot black-box optimization foundation model. |
| [Trace is the Next AutoDiff: Generative Optimization with Rich Feedback, Execution Traces, and LLMs](https://arxiv.org/pdf/2406.16218) | ArXiv | 2024 | Prompt-based | Replace traditional optimization trajectories with rich, structured execution traces. |
| [Visual Reasoning and Multi-Agent Approach in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP Combinatorial Challenges](https://arxiv.org/abs/2407.00092) | ArXiv | 2024 | Prompt-based | Utilize MLLMs to process 2D planar point distribution maps as input |
| [Revisiting OPRO: The Limitations of Small-Scale LLMs as Optimizers](https://arxiv.org/abs/2405.10276) | ACL | 2024 | Prompt-based | Discuss the model dependency of OPRO and identify its limitations on small-scale models. |
| [Exploring the True Potential: Evaluating the Black-box Optimization Capability of Large Language Models](https://arxiv.org/abs/2404.06290) | ArXiv | 2024 | Prompt-based | Conduct evaluation of LLMs on both discrete and continuous black-box optimization problems. |
| [LLMs can Schedule](https://arxiv.org/abs/2408.06993) | ArXiv | 2024 | Learning-based | Fine-tune LLMs with instruction-solution pairs for scheduling problems.|
| [An Optimizable Suffix Is Worth A Thousand Templates: Efficient Black-box Jailbreaking without Affirmative Phrases via LLM as Optimizer](https://arxiv.org/abs/2408.11313) |  NAACL | 2025 | Prompt-based | Apply iterative optimization to jailbreaking attack strategies. |
| [Generalists vs. Specialists: Evaluating LLMs on Highly-Constrained Biophysical Sequence Optimization Tasks](https://arxiv.org/abs/2410.22296) |  ArXiv | Apr 2025 | Learning-based | Integrate preference learning to train LLMs for satisfying complex biophysical constraints. |
| [Large Language Models for Combinatorial Optimization of Design Structure Matrix](https://arxiv.org/abs/2411.12571) | ArXiv | Nov 2024 | Prompt-based | Apply iterative optimization to design structure matrix sequencing. |
| [Large Language Model-Based Wireless Network Design](https://doi.org/10.1109/LWC.2024.3462556) | WCL | Sep 2024 | Prompt-based | Apply iterative optimization to wireless network design. |
| [Bridging Visualization and Optimization: Multimodal Large Language Models on Graph-Structured Combinatorial Optimization](https://arxiv.org/abs/2501.11968) |  ArXiv | 2025  | Prompt-based | Process visual representations of abstract graph structures using MLLMs. |
| [ORFS-agent: Tool-Using Agents for Chip Design Optimization](https://arxiv.org/abs/2506.08332) | arXiv | 2025 | Prompt-based | Apply iterative optimization to automated parameter tuning in chip design. |


### 3.2 Low-level LLMs for Optimization Algorithms

| **Title** | **Publication** | **Year** | **Type** | **Summary** |
|-----------|---------------|----------|----------|------------|
| [Language Model Crossover: Variation through Few-Shot Prompting](https://dl.acm.org/doi/abs/10.1145/3694791) | TELO | 2023 | Operators | Leverage LLMs as intelligent operators for textual genome crossover and recombination. |
| [GPT-NAS: Evolutionary Neural Architecture Search with the Generative Pre-Trained Model](https://arxiv.org/abs/2305.05351) | arXiv | 2023 | Initialization | Utilize LLMs for NAS initialization with prior knowledge. |
| [Large Language Models as Evolutionary Optimizers](https://ieeexplore.ieee.org/abstract/document/10611913) | CEC | 2023 | Operators | Employ LLMs as crossover, mutation, and selection operators to guide EAs. |
| [LLM Performance Predictors are good initializers for Architecture Search](https://arxiv.org/abs/2310.16712) | arXiv | 2023 | Initialization | Utilize LLMs as performance predictors to assist initialization processes. |
| [Large Language Model for Multiobjective Evolutionary Optimization](https://link.springer.com/chapter/10.1007/978-981-96-3538-2_13) | EMO | 2023 | Operators |Empower MOEA/D with LLMs through zero-shot prompting as search operators in multi-objective optimization. |
| [Optimized Financial Planning: Integrating Individual and Cooperative Budgeting Models with LLM Recommendations](https://www.mdpi.com/2673-2688/5/1/6) | AI | 2023 | Initialization | Assist non-expert users in initializing financial plans. |
| [Large Language Model-Based Evolutionary Optimizer: Reasoning with elitism](https://arxiv.org/abs/2403.02054) | Neucom | 2024 | Operators | Utilize LLMs to guide individuals from dual pools for exploration and exploitation. |
| [Integrating genetic algorithms and language models for enhanced enzyme design](https://academic.oup.com/bib/article/26/1/bbae675/7945613?login=false) | BiB | 2024 | Initialization | Initialize genetic algorithms with LLMs to generate high-quality mutant pools for enzyme design. |
| [LLM Guided Evolution - The Automation of Models Advancing Models](https://dl.acm.org/doi/abs/10.1145/3638529.3654178) | GECCO | 2024 | Operators | Enhance the creativity and diversity of the LLM by introducing different role-based prompts for NAS. |
| [Large Language Model-Aided Evolutionary Search for Constrained Multiobjective Optimization](https://arxiv.org/abs/2405.05767) | ICIC | 2024 | Operators | Generate minimal solutions within populations using LLMs to reduce interaction costs. |
| [Large Language Models for Tuning Evolution Strategies](https://arxiv.org/abs/2405.10999) | arXiv | 2024 | Configuration | Apply LLM-based feedback loops to sequentially optimize evolution strategies through static tuning. |
| [Large language models as surrogate models in evolutionary algorithms: A preliminary study](https://www.sciencedirect.com/science/article/abs/pii/S2210650224002797) | SWEVO | 2024 | Evaluation | Transform model-assisted selection tasks into classification and regression problems. |
| [LICO: Large Language Models for In-Context Molecular Optimization](https://arxiv.org/abs/2406.18851) | arXiv | 2024 | Evaluation | Apply LLMs as surrogate models in molecular science applications. |
| [Large Language Model-assisted Surrogate Modelling for Engineering Optimization](https://ieeexplore.ieee.org/abstract/document/10605321) | CAI | 2024 | Evaluation | Develop a collaborative framework using LLMs for model selection and training in engineering optimization. |
| [An investigation on the use of Large Language Models for hyperparameter tuning in Evolutionary Algorithms](https://dl.acm.org/doi/abs/10.1145/3638530.3664163) | GECCO | 2024 | Configuration | Implement step-size control for evolution strategies using OPRO-like mechanisms. |
| [Advancing Automated Knowledge Transfer in Evolutionary Multitasking via Large Language Models](https://arxiv.org/abs/2409.04270) | arXiv | 2024 | Operators | Utilize LLMs to assist mutation and other generative stages in evolutionary multi-task optimization. |
| [Large Language Model Aided Multi-objective Evolutionary Algorithm: a Low-cost Adaptive Approach](https://arxiv.org/abs/2410.02301) | arXiv | 2024 | Operators | Invoke LLMs for elite solution generation only upon insufficient population improvement. |
| [LLaMA Tunes CMA-ES](https://www.esann.org/sites/default/files/proceedings/2024/ES2024-136.pdf) | ESANN | 2024 | Configuration | Apply LLM-based feedback loops to sequentially optimize CMA-ES through dynamic control. |
| [Can Large Language Models Be Trusted as Evolutionary Optimizers for Network-Structured Combinatorial Problems?](https://arxiv.org/abs/2501.15081) | arXiv | 2025 | Operators | Validate LLM effectiveness in selection, crossover, and mutation phases while noting limitations in initialization |
| [LLM-Guided Evolution: An Autonomous Model Optimization for Object Detection](https://dl.acm.org/doi/abs/10.1145/3712255.3734340) | GECCO | 2025 | Operators | Utilize LLMs as crossover and mutation operators to optimize YOLO architectures for object detection. |
| [LAOS: Large Language Model-Driven Adaptive Operator Selection for Evolutionary Algorithms](https://dl.acm.org/doi/abs/10.1145/3712256.3726450) | GECCO | 2025 | Configuration | Utilize state features to replace optimization trajectories for guiding LLM-based operator selection. |
| [PAIR: A Novel Large Language Model-Guided Selection Strategy for Evolutionary Algorithms](https://arxiv.org/abs/2503.03239) | arXiv | 2025 | Operators | Focus on utilizing LLMs as selection operators to enhance LMEA. |
| [Large Language Model as Meta-Surrogate for Data-Driven Many-Task Optimization: A Proof-of-Principle Study](https://arxiv.org/abs/2503.08301) | arXiv | 2025 | Evaluation | Utilize LLMs as meta-Surrogates to facilitate cross-task knowledge sharing through token sequence representations. |
| [Large Language Models as Particle Swarm Optimizers](https://arxiv.org/abs/2504.09247) | arXiv | 2025 | Operators | Simulate PSO evolutionary processes using LLMs and refine LMEA for specific algorithms. |
| [Large Language Model-Driven Surrogate-Assisted Evolutionary Algorithm for Expensive Optimization](https://arxiv.org/abs/2507.02892) | arXiv | 2025 | Evaluation | Dynamically select appropriate proxy models and infill sampling criteria using LLMs. |

### 3.3 High-level LLMs for Optimization Algorithms


**Title**                                                                                                                                | **Publication** | **Year**   | **Summary**                                                                                                                                                                                                  



### 2. Low-level LLM-assisted Optimization Algorithms
#### 2.1 LLMs as Operators




#### 2.2 LLMs driven parameter setting problems
| **Title** | **Publication** | **Year** | **Summary** |
|-----------|--------------|------|----------|
| **Parameter Control** |
| [An investigation on the use of Large Language Models for hyperparameter tuning in Evolutionary Algorithms](https://dl.acm.org/doi/abs/10.1145/3638530.3664163) | GECCO | Aug 2024 | This study demonstrates the feasibility of using large language models (LLMs) like Llama2-70b for real-time hyperparameter tuning in evolutionary algorithms, specifically step-size adaptation in (1+1)-ES. Experimental results on BBOB benchmarks show that LLM-based strategies can rival traditional methods (e.g., one-fifth success rule), highlighting their potential as reasoning-driven optimizers for adaptive parameter control in complex optimization landscapes. |
| [LLaMA Tunes CMA-ES](https://www.esann.org/sites/default/files/proceedings/2024/ES2024-136.pdf) | ESANN | Oct 2024 | The paper proposes LLaMA-ES, a method that leverages the 70B-parameter LLaMA3 model to dynamically control CMA-ES hyperparameters through iterative analysis of optimization history. Experimental results on benchmark functions demonstrate significant performance improvements, showcasing the potential of LLMs in automating evolutionary algorithm adaptation. |
| [Metaheuristics and Large Language Models Join Forces: Toward an Integrated Optimization Approach](https://doi.org/10.1109/ACCESS.2024.3524176) | IEEE Access | Dec 2024 | This study proposes a novel integration of Large Language Models (LLMs) as pattern recognition tools within metaheuristics, enhancing combinatorial optimization by leveraging LLM-generated parameters (alpha and beta values) to guide node selection. The approach, validated on multi-hop influence maximization in social networks, demonstrates superior performance over existing methods, particularly for large-scale instances, by effectively combining LLM-driven insights with metaheuristic search. |
| **Parameter Tuning**  |
| [Large Language Models for Tuning Evolution Strategies](https://doi.org/10.48550/arXiv.2405.10999) | ArXiv | May 2024 | This paper proposes a feedback loop framework leveraging Large Language Models (LLMs) to autonomously tune Evolution Strategies (ES) parameters. By iteratively generating code, executing experiments, and analyzing results via LLMs (e.g., LLaMA3), the method effectively optimizes key parameters like τ, demonstrating improved performance on benchmark functions such as the 5D Sphere. The approach highlights LLMs’ potential in automating algorithm refinement and enabling self-adaptive optimization systems. |


#### 2.3 LLMs as Operators for Multi-objective Optimization
| **Title** | **Publication** | **Year** | **Summary** |
|-----------|--------------|------|----------|
| [Large Language Model for Multi-objective Evolutionary Optimization](https://arxiv.org/abs/2310.12541) | EMO | Oct 2023 | This study pioneers the use of large language models (LLMs) as zero-shot optimizers in multi-objective evolutionary optimization, decomposing problems into single-objective subproblems for LLM-driven solution generation without problem-specific training, and introduces a white-box linear operator inspired by LLM behavior to enhance efficiency and generalization across diverse optimization tasks. |
| [Large Language Model-Aided Evolutionary Search for Constrained Multiobjective Optimization](https://doi.org/10.1007/978-981-97-5581-3_18) | ICIC | May 2024 | This study integrates LLMs with evolutionary algorithms to solve constrained multi-objective optimization problems, where the LLM acts as a learnable search operator guided by tailored prompts incorporating objectives and constraints. Experimental results demonstrate that LLM-aided evolution significantly accelerates population convergence while maintaining diversity, outperforming state-of-the-art methods. |
| [Efficient Evolutionary Search Over Chemical Space with Large Language Models](https://arxiv.org/abs/2406.16976) — Molecular Design | ArXiv | Jun 2024 | MoLLEO integrates chemistry-aware large language models (LLMs) into evolutionary algorithms, acting as informed genetic operators to enhance molecular optimization. By leveraging LLMs like GPT-4 and BioT5 for crossover and mutation, it achieves superior performance in drug discovery tasks with fewer oracle calls, outperforming traditional methods in both single- and multi-objective settings. |
| [Large Language Model Aided Multi-objective Evolutionary Algorithm: a Low-cost Adaptive Approach](https://doi.org/10.48550/arXiv.2410.02301) | ArXiv | Oct 2024 | This study proposes a low-cost adaptive framework integrating large language models (LLMs) with multi-objective evolutionary algorithms (MOEAs), where LLMs act as black-box optimizers guided by elite solutions and structured prompts. The hybrid mechanism minimizes LLM interactions while leveraging its reasoning to accelerate convergence and enhance diversity, achieving superior performance on benchmark problems compared to state-of-the-art MOEAs. |
| [Large Language Model Based Multi-Objective Optimization for Integrated Sensing and Communications in UAV Networks](https://doi.org/10.1109/LWC.2025.3529082) | LWC | Oct 2024 | This study proposes LEDMA, an LLM-enabled evolutionary algorithm, to balance communication and sensing in UAV-enabled ISAC networks by jointly optimizing deployment and power control. By integrating LLMs as search operators with problem-specific prompts and decomposition-based optimization, LEDMA outperforms baseline methods in Pareto front quality and convergence, demonstrating the potential of LLMs in solving complex engineering MOPs. |
| [MOLLM: Multi-Objective Large Language Model for Molecular Design – Optimizing with Experts](https://arxiv.org/pdf/2502.12845) — Molecular Design | ArXiv | Feb 2025 | MOLLM is a novel LLM-driven framework for multi-objective molecular design, integrating in-context learning and genetic algorithms to optimize properties like drug-likeness and synthetic accessibility without task-specific training. It leverages LLMs as crossover/mutation operators and achieves state-of-the-art performance in efficiency and optimization quality, particularly excelling in complex scenarios with 5-6 objectives. |
| [Language Model Evolutionary Algorithms for Recommender Systems: Benchmarks and Algorithm Comparisons](https://arxiv.org/pdf/2411.10697) | ArXiv | Mar 2025 | This paper introduces LLM-enhanced evolutionary algorithms for optimizing recommendation prompts, demonstrating their potential through the RSBench benchmark while highlighting the synergy between large language models and evolutionary computation in multi-objective recommendation systems. |
| [An LLM-Empowered Adaptive Evolutionary Algorithm For Multi-Component Deep Learning Systems](https://doi.org/10.1609/aaai.v39i19.34303) | AAAI | Apr 2025 | This paper proposes ​​μMOEA​​, an LLM-enhanced evolutionary algorithm that leverages GPT-4's language understanding and reasoning to generate high-quality initial populations and escape local optima, significantly improving efficiency and diversity in detecting safety violations of multi-component deep learning systems (e.g., autonomous driving). |




#### 2.4 LLMs as Surrogate Models
| **Title** | **Publication** | **Year** | **Summary** |
|-----------|---------------|----------|------------|
| [Large Language Models as Surrogate Models in Evolutionary Algorithms: A Preliminary Study](https://arxiv.org/pdf/2406.10675) | SEC | Jun 2024 | This study proposes a novel LLM-based surrogate model for evolutionary algorithms, leveraging LLMs' inference capabilities to predict solution quality via zero-shot classification and regression tasks without training. Experimental results demonstrate that LLMs (e.g., Llama3 and Mixtral) achieve competitive performance compared to traditional surrogate models, offering a promising direction for reducing computational costs in expensive optimization problems. |
| [LICO: Large Language Models for In-Context Molecular Optimization](https://doi.org/10.48550/arXiv.2406.18851) | ArXiv | Jun 2024 | LICO extends pretrained LLMs for black-box molecular optimization by equipping them with domain-specific embedding layers and training on semi-synthetic functions (intrinsic properties + Gaussian Processes). It achieves state-of-the-art performance on the PMO benchmark through in-context learning, demonstrating LLMs' potential in scientific domains with limited data. The method highlights the critical role of language-guided embeddings and diversified training functions for generalization. |
| [Large Language Model-assisted Surrogate Modelling for Engineering Optimization](https://doi.org/10.1109/CAI59869.2024.00151) | CAI | Jun 2024 | This paper proposes a ChatGPT 4-assisted framework for selecting and training surrogate models in engineering optimization, demonstrating that LLM-generated workflows achieve performance comparable to conventional methods in both synthetic benchmarks and real-world automotive design tasks. The study highlights LLMs' potential to democratize data science expertise while noting limitations in extrapolation and data preprocessing. |

#### 2.5 LLMs for Bayesian Optimization
**Title** | **Publication** | **Year** | **Summary**
--- | --- | --- | ---
[Large Language Models to Enhance Bayesian Optimization](https://arxiv.org/pdf/2402.03921) | ICLR | Mar 2024 | LLAMBO integrates large language models (LLMs) with Bayesian optimization (BO) through natural language prompting, enhancing warm-starting, surrogate modeling, and candidate sampling via in-context learning. It achieves superior sample efficiency in hyperparameter tuning, particularly in low-data regimes, while maintaining modularity for seamless integration into existing BO frameworks.
[LLM-Enhanced Bayesian Optimization for Efficient Analog Layout Constraint Generation](https://doi.org/10.48550/arXiv.2406.05250) | ArXiv | Jun 2024 | LLANA introduces an LLM-enhanced Bayesian Optimization framework that leverages few-shot learning and contextual generation to efficiently explore analog layout constraints. Experiments demonstrate its superiority over GP and SMAC in sample efficiency and uncertainty calibration, enabling faster convergence with limited data.
[FunBO: Discovering Acquisition Functions for Bayesian Optimization with FunSearch](https://arxiv.org/pdf/2406.04824) | ArXiv | Jun 2024 | FunBO leverages LLMs to automatically discover interpretable acquisition functions via code evolution, achieving superior sample efficiency and generalization across diverse Bayesian optimization tasks, both in-distribution and out-of-distribution. This approach demonstrates the potential of combining evolutionary algorithms with large language models for algorithm design in mathematical sciences.
[Evolve Cost-aware Acquisition Functions Using Large Language Models](https://arxiv.org/pdf/2404.16906) | PPSN | Jun 2024 | The paper introduces EvolCAF, a framework integrating LLMs and evolutionary computation to automatically design cost-aware acquisition functions for Bayesian optimization. By leveraging LLMs for code generation and evolution, EvolCAF reduces expert dependency and achieves superior performance over human-crafted methods (e.g., EIpu, EI-cool) across synthetic and real-world tasks.
[ADO-LLM: Analog Design Bayesian Optimization with In-Context Learning of Large Language Models](https://arxiv.org/pdf/2406.18770) | ArXiv | Jun 2024 | ADO-LLM integrates Large Language Models (LLMs) with Bayesian Optimization (BO) to automate analog circuit design, leveraging LLMs' domain knowledge for rapid initial design generation and BO's exploration for diversity. By combining in-context learning with high-quality demonstrations from BO, the framework achieves superior efficiency and effectiveness, validated on multiple circuits with all specifications met.
[Searching for Optimal Solutions with LLMs via Bayesian Optimization](https://openreview.net/pdf?id=aVfDrl7xDV) | ICLR | Jan 2025 | BOPRO integrates Bayesian optimization with LLMs to dynamically balance exploration-exploitation in solution search. It outperforms baselines in word and molecule optimization but faces limitations in program synthesis due to coarse-grained code representations.
[Language-Based Bayesian Optimization Research Assistant (BORA)](https://arxiv.org/pdf/2501.16224) | ArXiv | Jan 2025 | BORA integrates large language models with Bayesian optimization to dynamically inject domain knowledge through hypothesis generation and real-time commentary. By adaptively switching between LLM-guided exploration and BO-driven exploitation, it achieves superior performance in high-dimensional scientific tasks, validated by synthetic benchmarks and real-world experiments in chemistry and energy.



### 3. High-level LLM-assisted Optimization Algorithms

#### 3.1 Algorithm Selection

| **Title**                                                                                      | **Publication** | **Year**  | **Summary**                                                                                                                                                                                                                                                                                               |
|--------------------------------------------------------------------------------------------|-------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [AS-LLM: When Algorithm Selection Meets Large Language Models](https://openreview.net/forum?id=l7aD9VMQUq) | Openreview  | Sep 2023 | This paper proposes an algorithm selection framework that integrates algorithm representation into the selection process, leveraging pre-trained Large Language Models (LLMs) for code comprehension. The approach demonstrates improved selection accuracy by using matching degree calculations between extracted algorithm and problem embeddings, and showcases the potential of LLMs in evaluating code representation. |
| [OptiMUS: Optimization Modeling Using MIP Solvers and Large Language Models](https://arxiv.org/abs/2310.06116) | ArXiv       | Oct 2023 | OptiMUS generates Python code from problem formulations to read input, call optimization solvers like Gurobi and cvxpy, and save the output. It utilizes advanced solver features and includes solver-specific instructions to correct common code errors and improve performance. |
| [Large Language Model-Enhanced Algorithm Selection: Towards Comprehensive Algorithm Representation](https://arxiv.org/abs/2311.13184) | IJCAI      | Nov 2023 | This paper introduces Large Language Models (LLMs) into the algorithm selection process for automated machine learning, bridging the gap in algorithm feature representation. The proposed model, which combines high-dimensional algorithm representations with problem features, outperforms existing techniques and provides valuable theoretical insights for practical implementation. |
| [OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models](https://doi.org/10.48550/arXiv.2402.10172) | ICML     | Feb 2024 | OptiMUS introduces a modular LLM-based agent framework that automates optimization modeling by integrating natural language processing with MILP solvers. It achieves state-of-the-art performance through connection graphs for context efficiency and multi-agent collaboration, outperforming prior methods by over 30% on complex datasets like NLP4LP. |
| [OptiMUS-0.3: Using Large Language Models to Model and Solve Optimization Problems at Scale](https://doi.org/10.48550/arXiv.2407.19633) | ArXiv     | Jul 2024 | OptiMUS-0.3 leverages modular LLM agents to automate optimization modeling, translating natural language problems into MILP formulations and solver code. It introduces self-corrective prompts, structure-aware modeling, and a novel dataset (NLP4LP), outperforming state-of-the-art methods by over 24% on complex tasks while enabling scalable human-in-the-loop validation. |

#### 3.2 Algorithm Generation

We further divide the "Algorithm Generation" process into two approaches: Single-round Generation and Iterative Generation. Single-round Generation involves generating a complete algorithm in a single interaction with the model, without further modifications. In contrast, Iterative Generation entails multiple rounds of refinement, where the model's output from each round serves as the input for the next, progressively improving the algorithm. 

#### 3.2.0 Benchmark and Plafrorm

| **Title** | **Publication** | **Year** |
|-------|-------------|------|
 [LLM4AD: A Platform for Algorithm Design with Large Language Model](https://arxiv.org/pdf/2412.17287) | ArXiv | Dec 2024 |
| [Beyond the Hype: Benchmarking LLM-Evolved Heuristics for Bin Packing](https://arxiv.org/pdf/2501.11411) | EvoApplications | Jan 2025 |
| [BLADE: Benchmark suite for LLM-driven Automated Design and Evolution of iterative optimisation heuristics](https://arxiv.org/pdf/2504.20183) | ArXiv | Apr 2025 |

#### 3.2.1 Single-round Generation

| **Title** | **Publication** | **Year** | **Summary** |
|-------|-------------|------|---------|
| [Leveraging Large Language Models for the Generation of Novel Metaheuristic Optimization Algorithms](https://dl.acm.org/doi/abs/10.1145/3583133.3596401) | GECCO | Jul 2023 | This paper explores the use of Large Language Models (LLMs) like GPT-4 to generate novel hybrid swarm intelligence optimization algorithms by decomposing six well-established algorithms for continuous optimization. The study focuses on the hybridization process, the challenges encountered, and the potential impact of LLM-generated algorithms in the metaheuristics field, offering insights into future research directions. |
| [Leveraging large language model to generate a novel metaheuristic algorithm with CRISPE framework](https://link.springer.com/article/10.1007/s10586-024-04654-6) | Cluster Computing | Apr 2024 | This paper introduces a novel metaheuristic algorithm, Zoological Search Optimization (ZSO), generated by ChatGPT-3.5 using the CRISPE framework, inspired by animal collective behaviors for continuous optimization. Experimental results on benchmark functions and engineering problems demonstrate the efficiency and effectiveness of ZSO-derived algorithms, highlighting the potential impact of Large Language Models (LLMs) in advancing the metaheuristics community. |

#### 3.2.2 Iterative Generation

| **Title** | **Publication** | **Year** | **Summary** |
|---|---|---|---|
| [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6) | Nature | Aug 2023 | FunSearch pioneers a novel AI-driven discovery paradigm by combining large language models with evolutionary algorithms to solve open mathematical problems and design optimized algorithms. This approach not only generates verifiable new knowledge in combinatorics (e.g., improved cap set constructions) but also produces human-interpretable programs that outperform classical heuristics in practical optimization challenges like online bin packing. |
| [Algorithm Evolution Using Large Language Model](https://arxiv.org/pdf/2311.15249) | ArXiv | Nov 2023 | AEL leverages LLMs within an evolutionary framework to automatically generate and optimize algorithms, demonstrating superior scalability and performance on TSP compared to human-designed heuristics and domain-specific models. By evolving algorithm-level strategies through crossover and mutation powered by LLMs, AEL reduces manual effort and achieves state-of-the-art generalization across problem scales. |
| [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model](https://arxiv.org/abs/2401.02051) | ICML | Jan 2024 | This paper introduces Evolution of Heuristics (EoH), an innovative method combining Large Language Models (LLMs) and Evolutionary Computation (EC) for Automatic Heuristic Design (AHD). EoH evolves both heuristic ideas (thoughts) in natural language and their corresponding executable codes, enhancing efficiency and effectiveness in generating high-performance heuristics. |
| [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf) | NeurIPS | Feb 2024 | ReEvo integrates large language models with reflective evolution to automate heuristic design for combinatorial optimization, where LLMs generate and refine heuristics through evolutionary search coupled with verbal feedback, achieving state-of-the-art performance across diverse problems with enhanced sample efficiency and generalization. |
| [AutoSAT: Automatically Optimize SAT Solvers via Large Language Models](https://arxiv.org/pdf/2402.10705) | ArXiv | Feb 2024 | AutoSAT automates the optimization of CDCL SAT solvers by leveraging LLMs to refine heuristic functions within a modular framework. It outperforms MiniSat on 9/12 datasets and surpasses Kissat on 4, demonstrating LLMs' potential in complex algorithm design through evolutionary-guided code generation. |
| [LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics](https://arxiv.org/abs/2405.20132) | TEVC | May 2024 | LLaMEA automates metaheuristic design by integrating large language models (LLMs) with evolutionary algorithms, where GPT-4 iteratively generates and refines optimization codes based on performance feedback from BBOB benchmarks. The framework produces novel algorithms (e.g., ERADS) that outperform state-of-the-art methods in 5D and show scalability to higher dimensions, demonstrating the potential of LLMs as modular algorithm synthesizers. |
| [L-AutoDA: Large Language Models for Automatically Evolving Decision-based Adversarial Attacks](https://arxiv.org/pdf/2401.15335) | GECCO | May 2024 | This paper introduces L-AutoDA, a framework that leverages large language models (LLMs) based on AEL (Algorithm Evolution Using Large Language Model) to automatically generate decision-based adversarial attack algorithms through evolutionary optimization, demonstrating superior performance over manual designs on CIFAR-10. |
| [Autonomous Multi-Objective Optimization Using Large Language Model](https://doi.org/10.48550/arXiv.2406.08987) | ArXiv | Jun 2024 | The paper proposes LLMOPT, a framework that leverages large language models (LLMs) to autonomously design evolutionary algorithm operators for multi-objective optimization. By integrating error-driven repair, dynamic selection, and LLM-powered crossover/mutation, LLMOPT generates robust operators that outperform traditional methods on diverse continuous and combinatorial optimization problems, reducing reliance on expert intervention. |
| [Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models](https://arxiv.org/abs/2407.10873) | PPSN | Jul 2024 | This study establishes the necessity of integrating evolutionary search with LLMs for automated heuristic design (AHD), demonstrating through large-scale benchmarks that standalone LLMs underperform evolutionary program search (EPS) methods like (1+1)-EPS and EoH across combinatorial optimization tasks, while revealing significant performance dependencies on problem types and LLM selection (e.g., code-specialized models outperform general-purpose ones in TSP). |
| [TS-EoH: An Edge Server Task Scheduling Algorithm Based on Evolution of Heuristic](https://arxiv.org/abs/2409.09063) | ArXiv | Sep 2024 | This paper presents a novel task-scheduling approach using Evolutionary Computing (EC) and heuristic algorithms to address the challenges of low-latency real-time processing in edge computing. By modeling service requests as task sequences and evaluating scheduling schemes through Large Language Models (LLMs), the proposed algorithm outperforms traditional methods, including heuristic and reinforcement learning approaches. |
| [Multi-objective Evolution of Heuristic Using Large Language Model](https://doi.org/10.48550/arXiv.2409.16867) | AAAI | Sep 2024 | The paper proposes MEoH, a multi-objective evolutionary framework leveraging large language models (LLMs) to automatically design heuristics that balance optimality and efficiency. By integrating dominance relationships in objective space and code dissimilarity via AST-based metrics, MEoH generates diverse non-dominated heuristics, achieving up to 10× speedup while maintaining competitive performance in combinatorial optimization tasks like bin packing and traveling salesman problems. |
| [In-the-loop Hyper-Parameter Optimization for LLM-Based Automated Design of Heuristics](https://arxiv.org/pdf/2410AAAI.16309) | ArXiv | Oct 2024 | The paper proposes LLaMEA-HPO, a hybrid framework integrating LLMs with hyper-parameter optimization (HPO), where LLMs focus on generating novel algorithmic structures while SMAC3 handles parameter tuning. This approach reduces LLM query costs by 80% and achieves state-of-the-art performance in heuristic design for problems like BBOB and TSP. |
| [Controlling the Mutation in Large Language Models for the Efficient Evolution of Algorithms](https://arxiv.org/pdf/2412.03250) | EvoApplications | Dec 2024 | This paper introduces ​​dynamic mutation prompts​​ with power-law-distributed rates to enhance LLM-driven algorithm evolution, showing that ​​GPT-4 outperforms GPT-3.5​​ in adhering to mutation controls and improving convergence speed in LLaMEA. |
| [HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs](https://arxiv.org/abs/2412.14995) | AAAI | Dec 2024 | This study explores the use of Large Language Models (LLMs) combined with evolutionary computation for automatic heuristic design (AHD), referred to as LLM-based Evolutionary Program Search (LLM-EPS). While previous methods like FunSearch, EoH, and ReEvo have shown promise, challenges remain in balancing exploration and exploitation in large heuristic search spaces. The authors propose two diversity measurement metrics to address this gap and introduce HSEvo, an adaptive framework that maintains this balance through a harmony search algorithm, achieving both high diversity and good objective scores efficiently. |
| [Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design](https://doi.org/10.48550/arXiv.2501.08603) | ArXiv | Jan 2025 | MCTS-AHD integrates Monte Carlo Tree Search with LLMs to automate heuristic design, enabling comprehensive exploration of heuristic spaces by retaining and refining underperforming candidates through tree-structured evolution. It outperforms population-based methods on NP-hard optimization tasks by balancing exploration-exploitation via decayed exploration factors and progressive widening. |
| [Complex LLM Planning via Automated Heuristics Discovery](https://arxiv.org/pdf/2502.19295) | ArXiv | Feb 2025 | AutoHD enables LLMs to generate explicit heuristic functions for planning tasks, which guide efficient search via algorithms like A* and evolve through iterative refinement. It achieves state-of-the-art performance across benchmarks (e.g., nearly doubled accuracy on some datasets) without additional training, offering both reliability and interpretability in complex reasoning. |
| [LLM-Assisted Automatic Memetic Algorithm for Lot-Streaming Hybrid Job Shop Scheduling With Variable Sublots](https://doi.org/10.1109/TEVC.2025.3556186) | TEVC | Mar 2025 | This paper proposes an LLM-assisted memetic algorithm (LLMMA) that automatically designs heuristics for solving complex lot-streaming hybrid job shop scheduling with variable sublots (LHJSV), demonstrating superior performance over traditional methods. |
| [Fitness Landscape of Large Language Model-Assisted Automated Algorithm Search](https://arxiv.org/pdf/2504.19636) | ArXiv | May 2025 | The paper introduces a ​​graph-based fitness landscape analysis​​ to characterize the behavior of ​​LLM-assisted algorithm search (LAS)​​, revealing its ​​multimodal, rugged nature​​ and task-dependent variations across different LLMs and search settings. This provides ​​practical insights​​ for optimizing automated algorithm design. |

### 4. Other Studies

#### 4.1 Autoformulation
| **Title** | **Publication** | **Year** | **Summary** |
|-----------|---------------|----------|------------|
| [OptiMUS: Optimization Modeling Using MIP Solvers and Large Language Models](https://arxiv.org/abs/2310.06116) | ArXiv | Oct 2023 | OptiMUS generates Python code from problem formulations to read input, call optimization solvers like Gurobi and cvxpy, and save the output. It utilizes advanced solver features and includes solver-specific instructions to correct common code errors and improve performance. |
| [LM4OPT: Unveiling the potential of Large Language Models in formulating mathematical optimization problems](https://www.tandfonline.com/doi/abs/10.1080/03155986.2024.2388452) | INFOR | Dec 2023 | This study evaluates GPT-4, GPT-3.5, and Llama-2-7b in translating natural language descriptions into mathematical optimization formulations, revealing GPT-4's superiority in one-shot settings (F1-score 0.63) and proposing the LM4OPT framework to enhance smaller models through progressive fine-tuning with noisy embeddings. While GPT models outperformed fine-tuned Llama-2-7b, the research highlights limitations of smaller LLMs in handling complex contexts compared to larger counterparts. |
| [OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models](https://doi.org/10.48550/arXiv.2402.10172) | ICML | Feb 2024 | OptiMUS introduces a modular LLM-based agent framework that automates optimization modeling by integrating natural language processing with MILP solvers. It achieves state-of-the-art performance through connection graphs for context efficiency and multi-agent collaboration, outperforming prior methods by over 30% on complex datasets like NLP4LP. |
| [OptiMUS-0.3: Using Large Language Models to Model and Solve Optimization Problems at Scale](https://doi.org/10.48550/arXiv.2407.19633) | ArXiv | Jul 2024 | OptiMUS-0.3 leverages modular LLM agents to automate optimization modeling, translating natural language problems into MILP formulations and solver code. It introduces self-corrective prompts, structure-aware modeling, and a novel dataset (NLP4LP), outperforming state-of-the-art methods by over 24% on complex tasks while enabling scalable human-in-the-loop validation. |
| [LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch](https://doi.org/10.48550/arXiv.2410.13213) | ICLR | Oct 2024 | The study proposes LLMOPT, a framework enhancing LLMs' generalization in solving diverse optimization problems (e.g., linear/nonlinear programming) by introducing a five-element formulation for unified problem representation and leveraging multi-instruction fine-tuning with model alignment. It achieves 11.08% higher accuracy than state-of-the-art methods across 20+ domains, demonstrating robust automation in optimization modeling and solving. |
| [Autoformulation of Mathematical Optimization Models Using LLMs](https://doi.org/10.48550/arXiv.2411.01679) | ArXiv | Nov 2024 | This paper introduces an LLM-enhanced framework for autoformulation of optimization models, leveraging Monte-Carlo Tree Search (MCTS) to systematically explore hierarchical components (variables, objectives, constraints) while pruning redundant hypotheses via SMT solvers. LLMs dynamically generate context-aware formulations and evaluate correctness, achieving superior performance in benchmarks through combined semantic and solver-based feedback. |
| [Enhancing healthcare resource allocation through large language models](https://doi.org/10.1016/j.swevo.2025.101859) | SEC | Nov 2024 | This study proposes an LLM-based framework integrating Prompt Engineering, RAG, and tool utilization to optimize healthcare resource allocation dynamically. ChatGPT-4o outperformed traditional methods by reducing costs and overtime while adapting to scale changes without re-optimization, demonstrating LLMs' potential in balancing multi-objective constraints and real-world medical complexities. |
| [Bridging Large Language Models and Optimization: A Unified Framework for Text-attributed Combinatorial Optimization](https://doi.org/10.48550/arXiv.2408.12214) | ArXiv | Dec 2024 | LNCS bridges LLMs and combinatorial optimization by encoding text-attributed problems into a unified semantic space and generating solutions via a Transformer-based network. Trained with conflict-free multi-task reinforcement learning, it achieves state-of-the-art performance across diverse COPs while enabling cross-problem generalization through fine-tuning. |
| [Meta-Optimization and Program Search using Language Models for Task and Motion Planning](https://arxiv.org/pdf/2505.03725) | ArXiv | May 2024 | The paper introduces ​​MOPS​​, a hierarchical method combining ​​language models​​ and ​​numerical optimization​​ to solve ​​task and motion planning (TAMP)​​ by searching over constraint sequences instead of action sequences, enabling precise robotic manipulation from natural language instructions. |
