# coding=utf-8

import argparse
import json
import random
from tqdm import tqdm

def main(args):
    random.seed(args.seed)
    f = open(args.output, mode="w")
    with open(args.input, mode="r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="processing"):
            data = json.loads(line.rstrip())["conversations"]
            # Select single round dialogue
            if len(data) == 2:
                assert data[0]["from"] == "human"
                assert data[1]["from"] == "assistant"
                human = data[0]["value"]
                assistant = data[1]["value"]
                if len(human) < 200:
                    p = random.random()
                    if p > 1 - args.sample_ratio:
                        json_data = dict(instruction=human, output=assistant)
                        f.write(json.dumps(json_data, ensure_ascii=False))
                        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_ratio", type=float, default=0.1)
    args = parser.parse_args()
    main(args)

