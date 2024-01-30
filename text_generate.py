# from luna.utils.activation_utils import get_llama_activations_bau, get_llama_loss, get_llama_probs
from luna.utils.prompter import Prompter
from luna.utils.llama import LLaMAForCausalLM, LLaMATokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import joblib
import logging
import json
import pandas as pd
"""
    function: parse_hidden_states
    description: parse the hidden states from the model output hidden states
    input: path to the hidden states, a list of layer indices
    output: a list of hidden states, a list of binary labels
"""
def parse_hidden_states(hidden_states, layer_indices=None):
    # trim the output of the embedding layer
    hidden_states = tuple(token[1:] for token in hidden_states)
    print(len(hidden_states), len(hidden_states[0]), hidden_states[0][0].shape)
    instance_result = None
    for i in range(len(hidden_states)):
        # traverse each token
        last_token_per_layer = None

        # For each tensor in the nested structure,
        # get the last element along the real token dimension
        for j in range(len(hidden_states[i])):
            if j not in layer_indices:
                continue
            # traverse each layer
            # hidden_states[i][j].shape = torch.Size([1, 73, 4096])
            if last_token_per_layer is None:
                last_token_per_layer = (
                    hidden_states[i][j][:, -1, :].detach().cpu().numpy()
                )

            else:
                last_token_per_layer = np.concatenate(
                    (
                        last_token_per_layer,
                        hidden_states[i][j][:, -1, :].detach().cpu().numpy(),
                    ),
                    axis=1,
                )

        if instance_result is None:
            instance_result = last_token_per_layer
        else:
            instance_result = np.concatenate(
                (instance_result, last_token_per_layer), axis=0
            )
    print("instance_result.shape", instance_result.shape)
    return instance_result


def truthful_qa_inference(
    fw,
    tokenizer,
    model,
    data_dict,
    device,
    max_new_tokens,
    extract_attention_block_idx,
    hidden_states_block_idx_list,
):
    max_new_tokens=70,
    for one_question in tqdm(data_dict):
        question_text = "Q: {}".format(one_question["question"])

        question_tokens = tokenizer.encode(question_text, return_tensors="pt").to(
            device
        )
        generated_tokens = model.generate(
            question_tokens,
            top_k=128,
            max_new_tokens=70,
            num_return_sequences=1,
            output_scores=True,
            output_hidden_states=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )
        # Get the hidden states
        current_hidden_states = generated_tokens.hidden_states
        hidden_states = parse_hidden_states(
            current_hidden_states, hidden_states_block_idx_list
        )

        len_question_tokens = len(question_tokens[0])
        generated_tokens = generated_tokens.sequences[0][len_question_tokens:]

        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("Generated text: {}".format(generated_text))

        question_tokens = question_tokens[0].tolist()

        step_by_step_layer_wise_activations = []
        step_by_step_head_wise_activations = []
        loss = []
        probs = []
        # for one_generated_token in generated_tokens.tolist():
        #     question_tokens.append(one_generated_token)

        #     # layer_wise_activations.shape = [num_blocks, num_tokens, hidden_size]
        #     # head_wise_activation.shape = [num_blocks, num_tokens, head_dim * num_of_head]
        #     (
        #         layer_wise_activations,
        #         head_wise_activations,
        #         _,
        #     ) = get_llama_activations_bau(
        #         model, torch.as_tensor([question_tokens]), device
        #     )

        #     step_by_step_layer_wise_activations.append(
        #         layer_wise_activations[extract_attention_block_idx + 1, -1, :]
        #     )
        #     step_by_step_head_wise_activations.append(
        #         head_wise_activations[extract_attention_block_idx, -1, :]
        #     )


        data_point = {
            "Q": question_text,
            "A": generated_text,
            "hidden_states": hidden_states,
            "attention_block_id": extract_attention_block_idx,
            "hidden_states_block_id": hidden_states_block_idx_list,
            "step_by_step_attention_heads": step_by_step_head_wise_activations,
            "step_by_step_attention_blocks": step_by_step_layer_wise_activations,
            "original_data_record": one_question,
        }

        joblib.dump(data_point, fw)


def advglue_inference(
    prompter,
    template,
    fw_dict,
    tokenizer,
    model,
    data_dict,
    device,
    max_new_tokens,
    extract_attention_block_idx,
    hidden_states_block_idx_list,
):
    '''
    answer_map = {
        "sst2": {"negative": 0, "positive": 1},
        "mnli": {"yes": 0, "maybe": 1, "no": 2},
        "mnli-mm": {"yes": 0, "maybe": 1, "no": 2},
        "qnli": {"yes": 0, "no": 1},
        "qqp": {"yes": 1, "no": 0},
        "rte": {"yes": 0, "no": 1},
    }
    '''
    for key in data_dict.keys():
        fw = fw_dict[key]
        for instance in tqdm(data_dict[key], desc="Generating for {}".format(key)):
            # ========= Handle special cases in mnli, qqp, and rte ========= #
            if "original_sentence" not in instance.keys():
                if key in ["mnli", "mnli-mm"]:
                    instance["sentence"] = "premise: {} hypothesis: {}".format(
                        instance["premise"], instance["hypothesis"]
                    )
                    if "original_hypothesis" in instance.keys():
                        instance["original_sentence"] = "premise: {} hypothesis: {}".format(
                            instance["premise"], instance["original_hypothesis"]
                        )
                    elif "original_premise" in instance.keys():
                        instance["original_sentence"] = "premise: {} hypothesis: {}".format(
                            instance["original_premise"], instance["hypothesis"]
                        )
                    else:
                        raise ValueError(
                            "mnli instance does not have premise and hypothesis"
                        )
                    
                elif key == "qqp":
                    instance["sentence"] = "question1: {} question2: {}".format(
                        instance["question1"], instance["question2"]
                    )
                    if "original_question2" in instance.keys():
                        instance["original_sentence"] = "question1: {} question2: {}".format(
                            instance["question1"], instance["original_question2"]
                        )
                    elif "original_question1" in instance.keys():
                        instance["original_sentence"] = "question1: {} question2: {}".format(
                            instance["original_question1"], instance["question2"]
                        )
                elif key == "rte":
                    if "original_sentence1" in instance.keys():
                        instance["sentence"] = "premise: {} hypothesis: {}".format(
                            instance["sentence1"], instance["sentence2"]
                        )
                        instance["original_sentence"] = "premise: {} hypothesis: {}".format(
                            instance["original_sentence1"], instance["sentence2"]
                        )
                    elif "original_sentence2" in instance.keys():
                        instance["sentence"] = "premise: {} hypothesis: {}".format(
                            instance["sentence1"], instance["sentence2"]
                        )
                        instance["original_sentence"] = "premise: {} hypothesis: {}".format(
                            instance["sentence1"], instance["original_sentence2"]
                        )

                elif key == "qnli":
                    if "question" in instance.keys() and "original_question" in instance.keys():
                        instance["sentence"] = "question: {} {}".format(
                            instance["question"], instance["sentence"]
                        )
                        instance["original_sentence"] = "question: {} {}".format(
                            instance["original_question"], instance["sentence"]
                        )
                    elif "sentence" in instance.keys() and "original_sentence" in instance.keys():
                        instance["sentence"] = "question: {} {}".format(
                            instance["question"], instance["sentence"]
                        )
                        instance["original_sentence"] = "question: {} {}".format(
                            instance["question"], instance["original_sentence"]
                        )

            # ========= Generate ========= #
            original_data, adv_data = None, None
            max_new_tokens=30,
            for k in instance.keys():
                if k == "original_sentence" or k == "sentence":
                    input_text = prompter.generate_prompt(template[key], instance[k])

                    input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(
                        device
                    )
                    generated_tokens = model.generate(
                        input_tokens,
                        top_k=128,
                        max_new_tokens=30,
                        num_return_sequences=1,
                        output_scores=True,
                        output_hidden_states=True,
                        output_attentions=True,
                        return_dict_in_generate=True,
                    )
                    # Get the hidden states
                    current_hidden_states = generated_tokens.hidden_states
                    hidden_states = parse_hidden_states(
                        current_hidden_states, hidden_states_block_idx_list
                    )

                    len_input_tokens = len(input_tokens[0])
                    # generated_tokens = generated_tokens.sequences[0]
                    generated_tokens = generated_tokens.sequences[0][len_input_tokens:]

                    generated_text = tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    generated_text = prompter.get_response(generated_text)
                    
                    print("Generated text: {}".format(generated_text))
                    
                    input_tokens = input_tokens[0].tolist()

                    step_by_step_layer_wise_activations = []
                    step_by_step_head_wise_activations = []
                    loss = []
                    probs = []

                    # for one_generated_token in generated_tokens.tolist():
                    #     input_tokens.append(one_generated_token)

                    #     (
                    #         layer_wise_activations,
                    #         head_wise_activations,
                    #         _,
                    #     ) = get_llama_activations_bau(
                    #         model, torch.as_tensor([input_tokens]), device
                    #     )

                    #     step_by_step_layer_wise_activations.append(
                    #         layer_wise_activations[
                    #             extract_attention_block_idx + 1, -1, :
                    #         ]
                    #     )
                    #     step_by_step_head_wise_activations.append(
                    #         head_wise_activations[extract_attention_block_idx, -1, :]
                    #     )
                    #     labels = []
                    #     for _ in range(len(input_tokens)):
                    #         labels.append(instance['label'])
                    #     label = tokenizer.encode(labels, add_special_tokens=False, return_tensors="pt").to(device)
                    #     loss.append(get_llama_loss(model, torch.as_tensor([input_tokens]), device, label))
                    #     probs.append(get_llama_probs(model, torch.as_tensor([input_tokens]), device))

                    data_point = {
                        "input": input_text,
                        "output": generated_text,
                        "label": instance["label"],
                        "is_adversarial": 0 if k == "original_sentence" else 1,
                        "hidden_states": hidden_states,
                        "attention_block_id": extract_attention_block_idx,
                        "hidden_states_block_id": hidden_states_block_idx_list,
                        "step_by_step_attention_heads": step_by_step_head_wise_activations,
                        "step_by_step_attention_blocks": step_by_step_layer_wise_activations,
                        "original_data_record": instance,
                        "loss": loss,
                        "probs": probs,
                        "is_attack_success": 1
                    }

                    if k == "original_sentence":
                        original_data = data_point
                    else:
                        adv_data = data_point
            
            if original_data['output'] != adv_data['output']:
                joblib.dump(original_data, fw)
                joblib.dump(adv_data, fw)
            else:
                original_data['is_attack_success'] = 0
                adv_data['is_attack_success'] = 0
                joblib.dump(original_data, fw)
                joblib.dump(adv_data, fw)
            

def ood_inference(
    template,
    fw_dict,
    tokenizer,
    model,
    data_dict,
    device,
    max_new_tokens,
    extract_attention_block_idx,
    hidden_states_block_idx_list,
):
    for key in data_dict.keys():
        fw = fw_dict[key]
        for instance in tqdm(data_dict[key], desc="Generating for {}".format(key)):
            prompter = Prompter()
            input_text = prompter.generate_prompt(template["sst2"], instance["sentence"])

            input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
            generated_tokens = model.generate(
                input_tokens,
                top_k=128,
                max_new_tokens=10,
                num_return_sequences=1,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            # Get the hidden states
            current_hidden_states = generated_tokens.hidden_states
            hidden_states = parse_hidden_states(
                current_hidden_states, hidden_states_block_idx_list
            )

            len_input_tokens = len(input_tokens[0])
            generated_tokens = generated_tokens.sequences[0][len_input_tokens:]

            generated_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            print("Generated text: {}".format(generated_text))

            input_tokens = input_tokens[0].tolist()

            step_by_step_layer_wise_activations = []
            step_by_step_head_wise_activations = []
            loss = []
            probs = []

            # for one_generated_token in generated_tokens.tolist():
            #     input_tokens.append(one_generated_token)

            #     (
            #         layer_wise_activations,
            #         head_wise_activations,
            #         _,
            #     ) = get_llama_activations_bau(
            #         model, torch.as_tensor([input_tokens]), device
            #     )

            #     step_by_step_layer_wise_activations.append(
            #         layer_wise_activations[extract_attention_block_idx + 1, -1, :]
            #     )
            #     step_by_step_head_wise_activations.append(
            #         head_wise_activations[extract_attention_block_idx, -1, :]
            #     )
                
            #     labels = []
            #     for _ in range(len(input_tokens)):
            #         labels.append(instance['label'])
            #     label = tokenizer.encode(labels, add_special_tokens=False, return_tensors="pt").to(device)
            #     loss.append(get_llama_loss(model, torch.as_tensor([input_tokens]), device, label))
            #     probs.append(get_llama_probs(model, torch.as_tensor([input_tokens]), device))

            data_point = {
                "input": input_text,
                "output": generated_text,
                "label": instance["label"],
                "is_ood": 1 if key != "dev" else 0,
                "hidden_states": hidden_states,
                "hidden_states_block_id": hidden_states_block_idx_list,
                "attention_block_id": extract_attention_block_idx,
                "step_by_step_attention_heads": step_by_step_head_wise_activations,
                "step_by_step_attention_blocks": step_by_step_layer_wise_activations,
                "loss": loss,
                "probs": probs,
                "original_data_record": instance,
            }

            joblib.dump(data_point, fw)


def main():
    """
    Specify dataset name as the first command line argument. Current options are
    "truthful_qa", "advglue++", "sst2". Gets activations for all prompts in the
    validation set for the specified dataset on the last token for llama-7B.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="alpaca_7B")
    parser.add_argument("--dataset_name", type=str, default="truthful_qa")
    parser.add_argument("--result_save_path", type=str, default="outputs")
    parser.add_argument("--extract_attention_block_idx", type=int, default=0)
    parser.add_argument(
        "--hidden_states_block_idx", type=str, default="31", help="example: 0_15_31"
    )
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, required=True)
    args = parser.parse_args()

    HF_NAMES = {
        "llama_7B": "decapoda-research/llama-7b-hf",
        "llama2_7B": "meta-llama/Llama-2-7b-hf",
        "alpaca_7B": "circulus/alpaca-7b",
        "vicuna_7B": "AlekseyKorshuk/vicuna-7b",
    }

    model_name = args.model_name
    MODEL = HF_NAMES[model_name]


    # if args.model_name == "llama2_7B":
    #     print("To Be Implemented...")
    #     raise NotImplementedError("llama2_7B is not supported yet!")
    #     # tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    #     # model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    # else:

    # tokenizer = LLaMATokenizer.from_pretrained(MODEL, cache_dir=args.cache_dir)
    # model = LLaMAForCausalLM.from_pretrained(
    #     MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto", cache_dir=args.cache_dir
    # )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto", cache_dir=args.cache_dir
    )

    device = args.device
    device = "cuda"
    max_new_tokens = args.max_new_tokens
    result_save_path = args.result_save_path
    extract_attention_block_idx = args.extract_attention_block_idx
    hidden_states_block_idx_str = args.hidden_states_block_idx
    # hidden_states_block_idx_list = [
    #     int(i) for i in args.hidden_states_block_idx.split("_")
    # ]
    dataset = args.dataset_name
    # hidden_states_block_idx_list = [i for i in range(32)]
    # for i in range(32):
    #     dataset_folder = "{}/{}/{}".format(
    #         result_save_path, dataset, str(i)
    #     )
    #     if not os.path.exists(dataset_folder):
    #         os.makedirs(dataset_folder)
    hidden_states_block_idx_list = [
        int(i) for i in args.hidden_states_block_idx.split("_")
    ]
    dataset = args.dataset_name
    model_name = args.model_name

    dataset_folder = "{}/{}/{}".format(
        result_save_path, dataset, hidden_states_block_idx_str
    )
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    logging.basicConfig(
        filename="logs/{}_{}.log".format(model_name, dataset), level=logging.INFO
    )

    prompter = Prompter(model_name.split("_")[0])

    # ========= Load Dataset ========= #
    data_dict = None
    # ========= QA ========= #
    if dataset == "truthful_qa":
        file_path = "{}/{}/{}/{}.joblib".format(
            result_save_path, dataset, hidden_states_block_idx_str, model_name
        )

        # Check if file exists
        if os.path.isfile(file_path):
            # If it exists, delete it
            os.remove(file_path)
            print(f"File {file_path} has been deleted.")

        # hidden_states_folder = "{}/{}/{}_hidden_states".format(
        #     result_save_path, dataset, model_name
        # )
        # if not os.path.exists(hidden_states_folder):
        #     os.makedirs(hidden_states_folder)

        fw = open(file_path, "ab+")
        print("File {} has been created.".format(file_path))
        data_dict = load_dataset("truthful_qa", "generation")["validation"]
        truthful_qa_inference(
            fw,
            tokenizer,
            model,
            data_dict,
            device,
            max_new_tokens,
            extract_attention_block_idx,
            hidden_states_block_idx_list,
        )

    # ========= ADV ========= #
    elif dataset == "advglue++":
        # read the template from dataset/adv_rob/prompt.json
        template = None
        with open("dataset/advglue++/prompt.json", "r") as f:
            template = json.load(f)

        # Initialize an empty dictionary to store the file writers
        fw_dict = {}

        # Loop through the file list
        data_dict = None
        with open("dataset/advglue++/data/alpaca.json", "r") as f:
            data_dict = json.load(f)

        # Check if data_dict is None or template is None
        if data_dict is None or template is None:
            raise ValueError("Dataset or template is None")

        # Create a file writer for each file
        for key in data_dict.keys():
            file_path = "{}/{}/{}/{}_{}.joblib".format(
                result_save_path, dataset, hidden_states_block_idx_str, model_name, key
            )

            # Check if file exists
            if os.path.isfile(file_path):
                # If it exists, delete it
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

            fw = open(file_path, "ab+")
            fw_dict[key] = fw
            print("File {} has been created.".format(file_path))

        advglue_inference(
            prompter,
            template,
            fw_dict,
            tokenizer,
            model,
            data_dict,
            device,
            max_new_tokens,
            extract_attention_block_idx,
            hidden_states_block_idx_list,
        )

    # ========= OOD ========= #
    elif dataset == "sst2":
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        # read template from dataset/ood/sst2/prompt.json
        template = None
        with open("dataset/sst2/prompt.json", "r") as f:
            template = json.load(f)

        # Define the path where the TSV files are stored
        path = "dataset/sst2/data"

        # Get a list of all TSV files in the specified directory
        file_list = [f for f in os.listdir(path) if f.endswith(".tsv")]

        # Initialize an empty dictionary to store the data
        data_dict = {}
        # Initialize an empty dictionary to store the file writers
        fw_dict = {}

        # Loop through the file list
        for file in file_list:
            # Read the TSV file into a DataFrame
            df = pd.read_csv(os.path.join(path, file), sep="\t")
            # Get the file name without the '.tsv' extension
            file_name, _ = os.path.splitext(file)
            # Convert the DataFrame to a dictionary and add it to 'data'
            data_dict[file_name] = df.to_dict(orient="records")

        # Check if data_dict is empty or template is None
        if data_dict == {} or template is None:
            raise ValueError("Dataset or template is None")

        # Create a file writer for each file
        for key in data_dict.keys():
            file_path = "{}/{}/{}/{}_{}.joblib".format(
                result_save_path, dataset, hidden_states_block_idx_str, key, model_name
            )

            # Check if file exists
            if os.path.isfile(file_path):
                # If it exists, delete it
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

            fw = open(file_path, "ab+")
            fw_dict[key] = fw
            print("File {} has been created.".format(file_path))

        ood_inference(
            template,
            fw_dict,
            tokenizer,
            model,
            data_dict,
            device,
            max_new_tokens,
            extract_attention_block_idx,
            hidden_states_block_idx_list,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
