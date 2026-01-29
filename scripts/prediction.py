import argparse
import csv
import json
import logging
import os
import pathlib
import sys

wd = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(wd))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, LassoLarsCV, RidgeCV

from lib.finetune.model.esm2 import FinetunedEsmModel, PretrainedEsmModel
from lib.predict.model.embedding import get_cls_reps, get_outputs
from lib.predict.model.score import (precision_top1to10,
                                     precision_top5_and_top10,
                                     recall_top1to10,
                                     recall_top5_and_top10)
from lib.utils.jsonl import dump_jsonl, read_jsonl
from lib.utils.logger import get_logger

logging.basicConfig(level=logging.INFO)


def main(args):
    # Protein language model
    if args.plm_type == "pretrained":
        plm = PretrainedEsmModel(
            model_name=f"facebook/{args.plm_name}",
        )
        info = f"{args.plm_name.replace('facebook/', '')}"
    if args.plm_type == "finetuned":
        plm = FinetunedEsmModel(
            model_name=f"facebook/{args.plm_name}",
            checkpoint_path=os.path.join(
                args.finetuned_model_path, args.finetuned_model_name
            ),
            remove_suffix=args.remove_suffix,
        )
        info = f"{os.path.dirname(os.path.dirname(args.finetuned_model_name))}/{os.path.basename(args.finetuned_model_name).replace('.pth', '')}"

    save_dir = os.path.join(args.log_dir, args.plm_type, info)
    os.makedirs(save_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.log_dir, args.plm_type, info, "args.log"))
    logger.info(json.dumps({"args": vars(args)}, indent=4, separators=(",", ": ")))

    score_summary = []
    dir_names = [
        dir_name
        for dir_name in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, dir_name))
    ]

    for k, dir_name in enumerate(sorted(dir_names)):
        # Dataset
        train_df = pd.DataFrame(
            read_jsonl(os.path.join(args.dataset_dir, f"{dir_name}/train.jsonl"))
        )
        valid_df = pd.DataFrame(
            read_jsonl(os.path.join(args.dataset_dir, f"{dir_name}/valid.jsonl"))
        )

        X_valid = get_cls_reps(get_outputs(valid_df.sequence.tolist(), plm))
        y_valid = valid_df.property.to_numpy()

        # Train and predict
        y_preds = []

        for _, rows in train_df.groupby("plate"):
            # Prediction model
            if args.model_name == "ridge":
                model = RidgeCV(
                    alphas=np.logspace(-6, 6, 1000),
                    gcv_mode="auto",
                    store_cv_values=True,
                )
            if args.model_name == "lasso":
                model = LassoCV(alphas=np.logspace(-6, 6, 1000), cv=3)
            if args.model_name == "lassolars":
                model = LassoLarsCV(n_jobs=-1, cv=3)

            # Train model
            X_train = get_cls_reps(get_outputs(rows.sequence.tolist(), plm))
            y_train = rows.property.to_numpy()
            model.fit(X_train, y_train)

            # Predict valid data
            y_pred = model.predict(X_valid)
            y_preds.append(y_pred.reshape(1, -1))

        ensembled_y_pred = np.concatenate(y_preds, axis=0).mean(axis=0)

        # Save estimates
        result = np.concatenate(
            [ensembled_y_pred.reshape(-1, 1), y_valid.reshape(-1, 1)], axis=1
        )
        estimates_save_dir = os.path.join(save_dir, "estimates", args.model_name)
        os.makedirs(estimates_save_dir, exist_ok=True)
        with open(os.path.join(estimates_save_dir, f"estimate_k{k}.csv"), "w") as f:
            dict_writer = csv.DictWriter(f, fieldnames=["estimate", "actual"])
            dict_writer.writeheader()
            for row in result:
                dict_writer.writerow({"estimate": row[0], "actual": row[1]})

        # Compute scores
        r = stats.pearsonr(y_valid, ensembled_y_pred).statistic
        rho = stats.spearmanr(y_valid, ensembled_y_pred).statistic
        precision_top5, precision_top10 = precision_top5_and_top10(
            ensembled_y_pred, y_valid
        )
        recall_top5, recall_top10 = recall_top5_and_top10(ensembled_y_pred, y_valid)

        score_summary.append(
            {
                "k": k,
                "n_valid": len(y_valid),
                "pearsonr": r,
                "spearmanr": rho,
                "precision_top5": precision_top5,
                "precision_top10": precision_top10,
                "recall_top5": recall_top5,
                "recall_top10": recall_top10,
            }
        )

        # Precision and Recall
        precisions = precision_top1to10(ensembled_y_pred, y_valid)
        recalls = recall_top1to10(ensembled_y_pred, y_valid)
        precision_recall_save_dir = os.path.join(
            save_dir, "precision_recall", args.model_name
        )
        os.makedirs(precision_recall_save_dir, exist_ok=True)
        with open(
            os.path.join(precision_recall_save_dir, f"precision_recall_k{k}.csv"), "w"
        ) as f:
            dict_writer = csv.DictWriter(f, fieldnames=["k", "presicion", "recall"])
            dict_writer.writeheader()
            for i, (v1, v2) in enumerate(zip(precisions, recalls), start=1):
                dict_writer.writerow({"k": i, "presicion": v1, "recall": v2})

    dump_jsonl(
        score_summary,
        os.path.join(
            args.log_dir, args.plm_type, info, f"scores_{args.model_name}.jsonl"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plm_name",
        type=str,
        choices=[
            "esm2_t6_8M_UR50D",
            "esm2_t12_35M_UR50D",
            "esm2_t30_150M_UR50D",
            "esm2_t33_650M_UR50D",
        ],
    )
    parser.add_argument("--plm_type", type=str, choices=["pretrained", "finetuned"])
    parser.add_argument("--finetuned_model_path", default="", type=str)
    parser.add_argument("--finetuned_model_name", default="", type=str)
    parser.add_argument(
        "--model_name",
        default="ridge",
        type=str,
        choices=["ridge", "lasso", "lassolars"],
    )
    parser.add_argument("--remove_suffix", default="module.", type=str)
    parser.add_argument("--dataset_dir", default="data/cytotoxicity/kfolds", type=str)
    parser.add_argument("--log_dir", default="results/prediction", type=str)
    args = parser.parse_args()
    main(args)
