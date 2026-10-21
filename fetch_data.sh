#!/usr/bin/env bash
# Fetch the public datasets and detector rules at the commits used in the paper.
set -euo pipefail
mkdir -p data && cd data
get() { git clone -q "$1" "$2" && git -C "$2" checkout -q "$3"; }
get https://github.com/verazuo/jailbreak_llms.git jailbreak_llms 4f4031b
get https://github.com/HumanCompatibleAI/tensor-trust-data.git tensor-trust-data 747a75e
get https://github.com/microsoft/BIPIA.git BIPIA a004b69
get https://github.com/liu00222/Open-Prompt-Injection.git Open-Prompt-Injection fbed5cc
get https://github.com/tatsu-lab/stanford_alpaca.git stanford_alpaca 761dc5b
get https://github.com/protectai/rebuff.git rebuff 4d2fe06
get https://github.com/deadbits/vigil-llm.git vigil-llm f2b4c13
