import random
from argparse import ArgumentParser

import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

from summarization.data.big_patent import BigPatentDataset
from summarization.lsa import LsaSummarizer
from summarization.models.longformer_pegasus import LongformerPegasusSummarizer
from summarization.models.pegasus import PegasusSummarizer


def generate_text(model, text, max_length=6144, device=None):
    if device is not None:
        device = torch.device(device)

    input = model.tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        return_attention_mask=True,
        return_tensors="pt",
    )

    if device is not None:
        input["input_ids"] = input["input_ids"].to(device)
        input["attention_mask"] = input["attention_mask"].to(device)

    beam_outputs = model.generate(
        input["input_ids"],
        attention_mask=input["attention_mask"],
        max_length=256,
        num_beams=5,
        repetition_penalty=5.0,
        # length_penalty=0.85,
        # num_return_sequences=3,
        early_stopping=True,
    ).cpu()
    output = [
        model.tokenizer.decode(beam_output, skip_special_tokens=True)
        for beam_output in beam_outputs
    ]
    return output[0]


def evaluate(args):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    og_model: PegasusSummarizer = PegasusSummarizer(
        pretrained_name=args.pegasus_pretrained_model
    )
    model: LongformerPegasusSummarizer = (
        LongformerPegasusSummarizer.load_from_checkpoint(
            args.longformer_pegasus_checkpoint
        )
    )

    og_model = og_model.to("cuda:0")
    model = model.to("cuda:1")
    lsa = LsaSummarizer()

    items = [
        item
        for cpc_code in ["a", "b", "c", "d", "e", "f", "g", "h", "y"]
        for item in BigPatentDataset.read_data("test", cpc_code)
    ]
    random.shuffle(items)
    if args.top_length_samples:
        samples = sorted(items, key=lambda x: len(x["abstract"]), reverse=True)[
            0 : args.num_samples
        ]
    else:
        samples = random.sample(items, args.num_samples)

    runs_og = []
    runs_us = []
    runs_lsa = []

    def run(sample_data, model, runs, device):
        description = sample_data["description"]
        abstract = sample_data["abstract"]
        generated = generate_text(model, description, max_length=1024, device=device)

        info = dict(
            abstract_length=len(abstract),
            description_length=len(description),
        )

        metrics = scorer.score(abstract, generated)
        runs.append(
            dict(
                **info,
                generated=generated,
                abstract=abstract,
                metrics=metrics,
                generated_length=len(generated),
            )
        )

    for sample_data in tqdm(samples):
        run(sample_data, og_model, runs_og)

        run(sample_data, model, runs_us)

        description = sample_data["description"]
        abstract = sample_data["abstract"]
        generated = lsa(description, 3)
        metrics = scorer.score(abstract, generated)
        info = dict(
            abstract_length=len(abstract),
            description_length=len(description),
        )
        runs_lsa.append(
            dict(
                **info,
                generated=generated,
                abstract=abstract,
                metrics=metrics,
                generated_length=len(generated),
            )
        )

    m_us = dict(
        rouge1=dict(recall=0, precision=0, fmeasure=0),
        rouge2=dict(recall=0, precision=0, fmeasure=0),
        rougeL=dict(recall=0, precision=0, fmeasure=0),
    )
    m_them = dict(
        rouge1=dict(recall=0, precision=0, fmeasure=0),
        rouge2=dict(recall=0, precision=0, fmeasure=0),
        rougeL=dict(recall=0, precision=0, fmeasure=0),
    )
    m_lsa = dict(
        rouge1=dict(recall=0, precision=0, fmeasure=0),
        rouge2=dict(recall=0, precision=0, fmeasure=0),
        rougeL=dict(recall=0, precision=0, fmeasure=0),
    )
    for run_us, run_og, run_lsa in zip(runs_og, runs_og, runs_lsa):
        for metric, value in run_us["metrics"].items():
            m_us[metric]["recall"] += value.recall
            m_us[metric]["precision"] += value.precision
            m_us[metric]["fmeasure"] += value.fmeasure

        for metric, value in run_og["metrics"].items():
            m_them[metric]["recall"] += value.recall
            m_them[metric]["precision"] += value.precision
            m_them[metric]["fmeasure"] += value.fmeasure

        for metric, value in run_lsa["metrics"].items():
            m_lsa[metric]["recall"] += value.recall
            m_lsa[metric]["precision"] += value.precision
            m_lsa[metric]["fmeasure"] += value.fmeasure

    length = len(runs_og)
    for metric in set([*m_us.keys(), *m_them.keys(), *m_lsa.keys()]):
        for name in ("recall", "precision", "fmeasure"):
            average_us = m_us[metric][name] / length
            average_og = m_them[metric][name] / length
            average_lsa = m_lsa[metric][name] / length
            print(
                f"[{metric} - {name}]: {average_us} us vs. {average_og} pegasus vs. {average_lsa} lsa"
            )


def main():
    parser = ArgumentParser(description="Evaluates Longformer-PEGASUS")
    parser.add_argument(
        "--longformer_pegasus_checkpoint",
        type=str,
        default="./pretrained/longformer-pegasus-bigpatent-best.ckpt",
        help="The name or path of the base model you want to convert",
    )
    parser.add_argument(
        "--pegasus_pretrained_model",
        type=str,
        default="google/pegasus-big_patent",
        help="The name or path of the base model you want to convert",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=25,
        help="random seed for test selection (-1 = don't set random seed)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="number of test samples",
    )
    parser.add_argument(
        "--top_length_samples", action="store_true", help="skip create long model"
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
