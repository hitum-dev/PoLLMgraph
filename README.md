#  PoLLMgraph: Unraveling Hallucinations in Large Language Models via State Transition Dynamics

Despite tremendous advancements in large language models (LLMs) over recent years,  a notably urgent challenge for their practical deployment is the phenomenon of  ``\textit{hallucination}'', where the model fabricates facts and produces non-factual statements. In response, we propose \texttt{PoLLMgraph}—a Polygraph for LLMs—as an effective model-based white-box detection and forecasting approach. \texttt{PoLLMgraph} distinctly differs from the large body of existing research that concentrates on addressing such challenges through black-box evaluations. In particular, we demonstrate that hallucination can be effectively detected by analyzing the LLM's internal state transition dynamics during generation via tractable probabilistic models. Experimental results on various open-source LLMs confirm the efficacy of \texttt{PoLLMgraph}, outperforming state-of-the-art methods by a considerable margin, evidenced by over 20\% improvement in AUC-ROC on common benchmarking datasets like TruthfulQA. Our work paves a new way for model-based white-box analysis of LLMs, motivating the research community to further explore, understand, and refine the intricate dynamics of LLM behaviors

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Function Overview](#function-overview)
- [Demo](#demo)

## Installation

### Setting up Python Environment

1. Ensure you have Python 3.8+ installed.
2. Clone this repository:
   ```bash
   git clone <repository-link>
3. Navigate to the project directory and set up a virtual environment:
   ```bash
   cd pollmgraph
   conda create -n env_name python=3.8
4. Activate the virtual environment:
   ```bash
   conda activate env_name
5. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt

### **Example**

### **Initializing the MetricsAppEvalCollections Object**

To evaluate the hallucination detection effectiveness, you first need to initialize the `MetricsAppEvalCollections` class. This class is responsible for collecting various metrics on evaluating the detection performance:

```python
eval_obj = MetricsAppEvalCollections(
    state_abstract_args_obj,
    prob_args_obj,
    train_instances,
    val_instances,
    test_instances,
)
```

Where:

- `state_abstract_args_obj`: A namespace object containing arguments related to state abstraction (e.g., dataset name, block index, info type).
- `prob_args_obj`: A namespace object containing arguments related to probability calculations (e.g., dataset, PCA dimension, model type).
- `train_instances`, `val_instances`, `test_instances`: The data instances for training, validation, and testing.

### **Collecting Metrics**

Once the `MetricsAppEvalCollections` object is initialized, you can then calculate various metrics. Here are some examples:

- Evaluating the model:

  ```python
  aucroc, accuracy, f1_score, _, _, hallucination_threshold = eval_obj.get_eval_result()
  ```

- Calculating entropy:

  ```python
  entropy = eval_obj.entropy()
  ```

---


## Datasets
We've conducted experiments on the following datasets:
---

### TruthfulQA Dataset Overview

- **Purpose**: TruthfulQA is designed to evaluate the truthfulness of Large Language Models (LLMs) in generating answers to questions.
- **Composition**: The dataset contains 817 questions across 38 categories of potential falsehoods, such as misconceptions and fiction.
- **Truth Assessment**: Answers' truthfulness is judged using fine-tuned GPT-3-13B models, classifying each response as true or false.

### Pre-requisites for Using the TruthfulQA Dataset

Before utilizing the TruthfulQA dataset, certain preparatory steps are required:

1. **Model Fine-Tuning**:
   - Follow the guide on [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://github.com/likenneth/honest_llama#truthfulqa-evaluation) to create GPT-JUDGE, a fine-tuned GPT-3 model.

2. **Dataset Preparation:**
   - Run the `add_scores_to_truthful_qa.py` script to process the dataset. 
   - Make sure to update the `file_name` and `file_with_score` variables in the script with the correct file paths.

   Execute the following command in your terminal:
   ```bash
   python add_scores_to_truthful_qa.py
   ```

## Function Overview

### **Model Abstraction**

The process of abstracting the behavior and properties of a system into a simplified representation that retains only the essential characteristics of the original system. In the context of this framework, model abstraction is done based on state and probabilistic models.

#### **1. ProbabilisticModel (from probabilistic_abstraction_model.py)**
- **Purpose**: Provides a base for creating probabilistic models based on abstracted states.
  
- **Usage Examples**:
  ```python
  # Initialize the ProbabilisticModel
  prob_model = ProbabilisticModel(args)
  
  # Evaluate LLM performance on a dataset task
  prob_model.eval_llm_performance_on_dataset_task()
  
  # Compose scores with ground truths
  prob_model.compose_scores_with_groundtruths_pair()
  ```

#### **2. AbstractStateExtraction (from state_abstraction_utils.py)**
- **Purpose**: Extracts abstract states from provided data instances.
  
- **Usage Examples**:
  ```python
  # Initialize the AbstractStateExtraction
  state_extractor = AbstractStateExtraction(args)
  
  # Perform PCA on data
  state_extractor.perform_pca()
  
  # (Additional method usage examples would be included if available in the file)
  ```

### **Metrics Calculation**

Metrics provide a quantitative measure to evaluate the performance and characteristics of models. In our framework, metrics evaluate the quality and behavior of abstracted models.

#### **1. MetricsAppEvalCollections (from metrics_appeval_collection.py)**
- **Purpose**: Acts as a central utility for metric evaluations based on state abstractions.
  
- **Usage Examples**:
  ```python
  # Initialize the MetricsAppEvalCollections
  metrics_evaluator = MetricsAppEvalCollections(args_obj1, args_obj2, train_data, val_data, test_data)
  
  # Retrieve evaluation results
  aucroc, accuracy, f1_score, _, _, _ = metrics_evaluator.get_eval_result()
  
  # Calculate the preciseness of predictions
  preciseness_mean, preciseness_max = metrics_evaluator.preciseness()
  ```


### **Demo**

To run a whole PoLLMGraph(MM) pipeline
   ```
   python demo.py 
   ```

