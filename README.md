# Email ticket classifier

This project demonstrates a complete workflow for email text classification in an industrial setting using an open-source large language model (~13B parameters). It covers 4-bit quantization, Unsloth - PEFT/LoRA fine-tuning, inference, and benchmarking. This classification module can function as the “classification agent” within a multi-agent system to automatically identify the category of incoming emails, preparing them for subsequent steps (e.g., specialized extraction agents).

Note: No original data or fine-tuned model weights are provided. We only share the scripts. Users are responsible for training the model with their own data and base model.


## Project Structure
```
.
├── README.md
├── classification_fine_tuning_classic_llm.ipynb
├── classification_fine_tuning_multimodal_model.ipynb
├── benchmark_ft_models.py
└── prediction_unsloth_ft_models.py
```

- classification_fine_tuning_classic_llm.ipynb <br/>
Example script for fine-tuning a pure text LLM (~13B parameters), showcasing 4-bit quantization and LoRA methods.

- classification_fine_tuning_multimodal_model.ipynb <br/>
Example script for fine-tuning a multimodal model (with a vision branch). Primarily freezes the vision portion and focuses on the text part.

- benchmark_ft_models.py <br/>
Benchmarking script to evaluate both pre-trained and fine-tuned models, generating metrics like classification accuracy and confusion matrices.

- prediction_unsloth_ft_models.py <br/>
Inference script demonstrating how to load fine-tuned weights and classify incoming email text.

## Key Features
- Unsloth: Fine-tuning and prediction steps are both powered by the Unsloth platform, providing an easier interface for quantized training, inference, and specialized LLM management.
- 4-bit Quantization: Reduces GPU memory footprint, enabling operation on resource-constrained environments.
- PEFT/LoRA: Fine-tunes only part of the model weights, greatly lowering training overhead.
- Email Text Classification: Assigns incoming emails to predefined categories with high accuracy.
- Extensible: This classification agent can integrate with other agents (e.g., extraction agents) in a multi-agent architecture.

## Dependencies
- Python 3.10+
- PyTorch (version depends on your GPU capability)
- Transformers
- Unsloth
- PEFT
- Other typical libraries (numpy, pandas, scikit-learn, etc.)
- If you use a multimodal model, ensure that additional dependencies (e.g., OpenCV, Pillow) are installed if needed.

## Citation
If you use or reference this work in your research or product, you may cite it as follows (example format): <br/>
```
@inproceedings{XxxEtAl2025,
title     = {Une architecture multi-agents pour la génération automatique de tickets en environnement industriel : focus sur l’agent de classification},
author    = {Xxx, Yyy, and Zzz},
booktitle = {xxx 2025},
year      = {2025},
url       = {x x x},
}
```
## License
This project is open-sourced under the MIT License. You are free to copy, modify, merge, and distribute the code. Please retain the original license notice.