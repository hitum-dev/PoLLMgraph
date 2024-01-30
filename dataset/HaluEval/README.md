# AdvGLUE++ Evaluation

`alpaca.json` contains the adversarial sample dataset, and prompt contains the prompts for different tasks.

There are in total five tasks:
```shell
SST-2: sentiment classification 
QQP: duplicate question detection
MNLI: (multi-genre) natural language inference 
QNLI: (question-answering) natural language inference 
RTE: natural language inference
```

with five attack method:
```shell
TextBugger
TextFooler
BERT-ATTACK
SememePSO
SemAttack
```

The answer mapping is as follows:

```shell
"sst2": {"negative": 0, "positive": 1},
"mnli": {"yes": 0, "maybe": 1, "no": 2},
"qnli": {"yes": 0, "no": 1},
"qqp": {"yes": 1, "no": 0},
"rte": {"yes": 0, "no": 1},
```

For each task, we should evaluate on `original_sentence` and `sentence` seperately and observe whether the outputs are different. If so the `sentence` should be an adversarial example.

If the LLM output something does not exist in the answer mapping, we can also record it as -1 `nonexistence rate`
