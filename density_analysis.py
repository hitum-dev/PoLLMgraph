from pollmgraph.metrics_appeval_collection import MetricsAppEvalCollections as Metrics
import pollmgraph.data_loader as data_loader

from types import SimpleNamespace
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import numpy as np
from scipy.stats import mannwhitneyu, rankdata
import os
import json
import argparse


def compute_A12(sample1, sample2):
    # Combine the samples
    combined = np.concatenate([sample1, sample2])

    # Rank the combined sample
    ranks = rankdata(combined)

    # Get the ranks for sample1
    ranks_sample1 = ranks[: len(sample1)]

    # Calculate R1, the sum of ranks for sample1
    R1 = np.sum(ranks_sample1)

    # Calculate A12
    m = len(sample1)
    n = len(sample2)
    A12 = (R1 / m - (m + 1) / 2) / n

    return A12


def rq1(state_abstract_args, prob_args, train_instances, val_instances, test_instances):
    state_abstract_args_obj = SimpleNamespace(**state_abstract_args)
    prob_args_obj = SimpleNamespace(**prob_args)

    metrics_obj = Metrics(
        state_abstract_args_obj,
        prob_args_obj,
        train_instances,
        val_instances,
        test_instances,
    )

    # Dictionary to store results
    results = {}
    # results["transition_gain"] = metrics_obj.transition_gain()
    results[
        "transition_KL_divergence_instance_level"
    ] = metrics_obj.transition_KL_divergence_instance_level()
    results[
        "transition_MMD_instance_level"
    ] = metrics_obj.transition_MMD_instance_level()
    results["transition_matrix_list"] = metrics_obj.transition_matrix_list()

    results[
        "stationary_distribution_entropy"
    ] = metrics_obj.stationary_distribution_entropy()

    return results


def load_data(state_abstract_args):
    args = SimpleNamespace(**state_abstract_args)
    llm_name = args.llm_name
    result_save_path = args.result_save_path
    dataset = args.dataset
    info_type = args.info_type
    extract_block_idx_str = args.extract_block_idx
    is_attack_success = args.is_attack_success

    dataset_folder_path = "{}/{}/{}".format(
        result_save_path, dataset, extract_block_idx_str
    )
    if not os.path.exists(dataset_folder_path):
        os.makedirs(dataset_folder_path)

    eval_folder_path = "eval/{}/{}".format(dataset, extract_block_idx_str)
    if not os.path.exists(eval_folder_path):
        os.makedirs(eval_folder_path)

    loader = None
    if dataset == "truthful_qa":
        loader = data_loader.TqaDataLoader(dataset_folder_path, llm_name)

    elif dataset == "advglue++":
        loader = data_loader.AdvDataLoader(
            dataset_folder_path, llm_name, is_attack_success
        )

    elif dataset == "sst2":
        loader = data_loader.OodDataLoader(dataset_folder_path, llm_name)

    else:
        raise NotImplementedError("Unknown dataset!")

    if info_type == "hidden_states":
        print("Loading hidden states...")
        (
            train_instances,
            val_instances,
            test_instances,
        ) = loader.load_hidden_states()
        print("Finished loading hidden states!")

    elif info_type == "attention_heads" or info_type == "attention_blocks":
        if info_type == "attention_heads":
            print("Loading attention heads...")
            (
                train_instances,
                val_instances,
                test_instances,
            ) = loader.load_attentions(0)
            print("Finished loading attention heads!")
        else:
            print("Loading attention blocks...")
            (
                train_instances,
                val_instances,
                test_instances,
            ) = loader.load_attentions(1)
            print("Finished loading attention blocks!")
    else:
        raise NotImplementedError("Unknown info type!")
    return train_instances, val_instances, test_instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_save_path",
        type=str,
        default="../../../data/llmAnalysis/songda",
        required=True,
    )
    parser.add_argument("--llm", type=str, default="alpaca_7B")

    args = parser.parse_args()

    datasets = {
        "truthful_qa": "GMM_400_2048_DTMC_0_0",
        "advglue++": "Grid_10_3_DTMC_0_1",
        "sst2": "GMM_200_1024_DTMC_0_0",
    }

    # datasets = {
    #     "humaneval": "GMM_400_1024_DTMC_0_0",
    #     "mbpp": "GMM_400_1024_DTMC_0_0"
    # }
    info_type = "hidden_states"
    fig_kde, axs_kde = plt.subplots(
        1, 3, figsize=(12, 3)
    )  # one row, three columns for KDE plots
    fig_kl, axs_kl = plt.subplots(
        1, 3, figsize=(12, 3)
    )  # one row, three columns for KL Divergence plots

    idx = 0
    result = {}
    for dataset, optimal_setting in datasets.items():
        settings = optimal_setting.split("_")
        state_abstract_args = {
            "llm_name": args.llm,
            "result_save_path": args.result_save_path,
            "dataset": dataset,
            "cluster_method": settings[0],
            "abstract_state": int(settings[1]),
            "test_ratio": 0.2,
            "extract_block_idx": "31",
            "info_type": info_type,
            "pca_dim": int(settings[2]),
            "is_attack_success": 1,
            "grid_history_dependency_num": int(settings[5]),
            "result_eval_path": "{}/eval/{}".format(args.result_save_path, args.llm),
        }

        prob_args = {
            "dataset": dataset,
            "cluster_method": settings[0],
            "abstract_state": int(settings[1]),
            "test_ratio": 0.2,
            "extract_block_idx": "31",
            "info_type": info_type,
            "pca_dim": int(settings[2]),
            "is_attack_success": 1,
            "hmm_components_num": int(settings[4]),
            "iter_num": 1000,
            "model_type": settings[3],
            "grid_history_dependency_num": int(settings[5]),
        }

        train_instances_loaded, val_instances_loaded, test_instances_loaded = load_data(
            state_abstract_args
        )

        train_instances = deepcopy(train_instances_loaded)
        val_instances = deepcopy(val_instances_loaded)
        test_instances = deepcopy(test_instances_loaded)

        results = rq1(
            state_abstract_args,
            prob_args,
            train_instances,
            val_instances,
            test_instances,
        )

        # transition_gain, normal_coverage_dict, abnormal_coverage_dict = results["transition_gain"]

        transition_distribution_divergence = results[
            "transition_KL_divergence_instance_level"
        ]
        transition_distribution_MMD = results["transition_MMD_instance_level"]
        train_probs, val_probs, test_probs = results["transition_matrix_list"]

        print(max(train_probs), max(test_probs), max(val_probs))
        print(min(train_probs), min(test_probs), min(val_probs))

        # Update the KDE plots
        num_samples = 10000
        train_probs = np.array(train_probs)
        test_probs = np.array(test_probs)
        val_probs = np.array(val_probs)
        # For test_probs
        sampled_indices_test = np.random.choice(
            np.arange(len(test_probs)), size=num_samples, p=test_probs / sum(test_probs)
        )

        sns.kdeplot(
            sampled_indices_test,
            fill=True,
            color="darkred",
            label="TestAbnormal",
            linewidth=2,
            ax=axs_kde[idx],
        )

        sampled_indices_val = np.random.choice(
            np.arange(len(val_probs)), size=num_samples, p=val_probs / sum(val_probs)
        )
        sns.kdeplot(
            sampled_indices_val,
            fill=True,
            color="darkblue",
            label="TestNormal",
            linewidth=2,
            ax=axs_kde[idx],
        )

        sampled_indices_train = np.random.choice(
            np.arange(len(train_probs)),
            size=num_samples,
            p=train_probs / sum(train_probs),
        )
        sns.kdeplot(
            sampled_indices_train,
            fill=True,
            color="darkgreen",
            label="TrainNormal",
            linewidth=2,
            ax=axs_kde[idx],
        )
        dataset_perspective_dict = {
            "truthful_qa": "TruthfulQA",
            "sst2": "SST-2",
            "advglue++": "AdvGLUE++",
        }
        axs_kde[idx].set_title(f"{dataset_perspective_dict[dataset]}", fontsize=16)
        # axs_kde[idx].set_xlabel('Index', fontsize=14)
        # axs_kde[idx].grid(True, linestyle='--', linewidth=0.5, color='gray')
        axs_kde[idx].set_xticks([])
        axs_kde[idx].set_xlim([0, None])
        if idx == 0:
            axs_kde[idx].set_ylabel("Density", fontsize="x-small")
            axs_kde[idx].legend(loc="upper left", fontsize="x-small")
        else:
            axs_kde[idx].set_ylabel("")
            axs_kde[idx].legend().remove()

        # Update the KL Divergence plots
        transition_distribution_divergence = np.array(
            transition_distribution_divergence
        )
        transition_distribution_divergence = transition_distribution_divergence[
            np.isfinite(transition_distribution_divergence)
        ]
        sns.kdeplot(transition_distribution_divergence, fill=True, ax=axs_kl[idx])

        axs_kl[idx].set_title(f"KL Divergence for {dataset}", fontsize=16)
        axs_kl[idx].set_xlabel("KL Divergence", fontsize=14)
        axs_kl[idx].set_ylabel("Density", fontsize=14)
        # axs_kl[idx].grid(True, linestyle='--', linewidth=0.5, color='gray')
        idx += 1

        # Mann-Whitney U-test
        u_statistic, p_value = mannwhitneyu(
            val_probs, test_probs, alternative="two-sided"
        )
        print(f"{dataset} U-statistic: {u_statistic}")
        print(f"{dataset} P-value: {p_value}")

        # Compute A12 effect size
        A12 = compute_A12(test_probs, val_probs)
        print(f"{dataset} A12: {A12}")

        result[dataset] = {
            "kde": {
                "train_probs": train_probs.tolist(),
                "test_probs": test_probs.tolist(),
                "val_probs": val_probs.tolist(),
            },
            "kl_divergence": transition_distribution_divergence.tolist(),
            "mannwhitneyu": {
                "u_statistic": u_statistic,
                "p_value": p_value,
            },
        }
    # Save plots
    fig_kde.tight_layout()
    fig_kde.savefig("eval/RQ1_KDE_comparison.png")
    fig_kl.tight_layout()
    fig_kl.savefig("eval/RQ1_KL_Divergence_comparison.png")
    # save the results to json
    with open("eval/RQ1.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
