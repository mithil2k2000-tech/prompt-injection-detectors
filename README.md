# Cheap prompt-injection detectors do not travel

📄 **Paper (PDF): [prompt-injection-detectors.pdf](prompt-injection-detectors.pdf)** · [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23593064.svg)](https://doi.org/10.5281/zenodo.23593064)

A cross-dataset and evasion study of lightweight prompt-injection detectors: Rebuff's heuristic and Vigil's YARA rules
as shipped, and logistic regression on word and character n-grams, on four public attack collections (JailbreakHub,
Tensor Trust, BIPIA, Open-Prompt-Injection; 7,239 prompts). Preprint, not peer reviewed.

* At shipped thresholds the heuristics caught at most 13% of attacks on three of four datasets. Rebuff's 0.75 threshold
  is above the score of "Ignore previous instructions." (0.70).
* N-gram classifiers: AUC 0.85–1.00 within a dataset, median 0.64–0.66 on another dataset.
* At a 1% false-positive rate, benign padding, leetspeak, zero-width characters or base64 cut the character n-gram
  classifier from 0.53 to 0.10 or less; Vigil detected nothing after any transformation.
* Rebuff's normalisation made it immune to zero-width characters and padding, by accident.

Only the cheap layers of Rebuff and Vigil were tested, not their model-based scanners.

## Reproduce

```bash
./fetch_data.sh
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python src/pi.py        # several hours on a CPU, mostly Rebuff; resumable
python src/analyse.py
cd paper && latexmk -pdf main.tex
```

## AI assistance

An AI assistant (Claude, Anthropic) helped write the experiment code and draft text. The author designed the study,
reviewed the code and results, and is responsible for the content.

## Licence

MIT for this repository's code. The datasets and detector rules keep their own licences and are not redistributed.
