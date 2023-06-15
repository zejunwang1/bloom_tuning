# coding=utf-8

import argparse
import torch
from transformers import BloomForCausalLM, BloomTokenizerFast

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path)
    model = BloomForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()
    
    print("基于bloom-560m的聊天机器人, quit/stop退出")
    input_pattern = "{}</s>"
    while True:
        text = input("用户: ")
        if text == "stop" or text == "quit":
            break

        input_ids = tokenizer(input_pattern.format(text), return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        outputs = model.generate(
            input_ids, 
            do_sample=True,
            max_new_tokens=args.max_new_tokens, 
            top_p=args.top_p, 
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty, 
            eos_token_id=tokenizer.eos_token_id
        )
        
        input_ids_len = input_ids.size(1)
        response_ids = outputs[0][input_ids_len:]
        response = tokenizer.decode(response_ids)
        print("Assistant: {}\n".format(response.strip().replace('</s>', "")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    args = parser.parse_args()
    main(args)

