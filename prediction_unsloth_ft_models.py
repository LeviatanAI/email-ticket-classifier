# -*- coding: utf-8 -*-
"""
Example usage:
--------------
Before running this script, make sure you have installed the required packages:
    !pip install unsloth
    !pip install -U bitsandbytes

You can then run this script with:
    python prediction_unsloth_ft_models.py

Description:
------------
This script loads a fine-tuned language or multimodal model (Qwen or Llama),
reads a CSV file containing text samples to classify, and produces categorization
predictions. The script is designed to categorize tickets based on the
description provided.

The categories are in French, but the script logic and docstrings are in English
following NumPy style guidelines. This script serves as an example of how to use
the `unsloth` library with a fine-tuned model for inference.

Author:
-------
Leviatan | AI research lab

"""

import os
import time
import pandas as pd
from unsloth import FastLanguageModel, FastVisionModel


###############################################################################
# Mutable environment variables
###############################################################################
MODEL_NAME = "Qwen-2.5-14B-FT-1ep"
#MODEL_NAME = "Qwen-2.5-14B-FT-30step"
#MODEL_NAME = "Llama-3.2-11B-FT-1ep"
#MODEL_NAME = "Llama-3.2-11B-FT-30step"

# Change the MAIN_PATH variable to match the main path specific to your environment
# The test_set_filtred.csv should be in the MAIN_PATH folder
MAIN_PATH = "home/ubuntu"

# False for using the entire dataset, True for using a small 20-sample dataset for testing
IS_DEV_TEST = False

# Leviatan Hugging Face token
HF_TOKEN = "hf_1234567"

MAX_SEQ_LENGTH = 2048

###############################################################################
# Constants
###############################################################################
MODELS = {
    "Llama-3.2-11B-FT-1ep": {
        "model_path": "Model_Path_Placeholder",
        "model_type": "multimodal",
        "chat_template": "llama"
    },
    "Llama-3.2-11B-FT-30step": {
        "model_path": "Model_Path_Placeholder",
        "model_type": "multimodal",
        "chat_template": "llama"
    },
    "Qwen-2.5-14B-FT-1ep": {
        "model_path": "Model_Path_Placeholder",
        "model_type": "instruct",
        "chat_template": "qwen"
    },
    "Qwen-2.5-14B-FT-30step": {
        "model_path": "Model_Path_Placeholder",
        "model_type": "instruct",
        "chat_template": "qwen"
    },
}

# Prompt in French for ticket classification
PROMPT_TEMPLATE = """
Tu es un assistant spécialisé dans la classification de tickets à partir de leur contenu textuel. Ton objectif est d’analyser la description fournie et de l’associer à l’une des catégories ci-dessous :
```
Demande de service/Backup #BCS/Autre
Demande de service/Backup #BCS/Demande de renseignement
Demande de service/Backup #BCS/Restauration qualifiée
Demande de service/Backup #BCS/Stratégie de sauvegarde/Création
Demande de service/Backup #BCS/Stratégie de sauvegarde/Modification
Demande de service/Backup #BCS/Stratégie de sauvegarde/Suppression
Demande de service/Cyber Sécurité #CS2/Bastion/Création-Modification d'entrées
Incidents/Backup #BCS/Sauvegarde
Incidents/Supervision
```
### Règles :
1. Réponds uniquement par la catégorie exacte, sans texte supplémentaire.
2. La catégorie associée doit être unique.
3. Si aucune catégorie ne correspond, la valeur de "categorisation" doit être "unknown".

### Exemple :
La description:
Bonjour¤ Merci de relancer les sauvegardes FULL¤ si ils ne sont pas repassées en automatique. Ghislain Envoyé de mon iPhone Début du message transféré
#### Sortie :
{{
  "categorisation": "Incidents/Backup #BCS/Sauvegarde"
}}

Analyse uniquement la description suivante :
{content}
"""


def build_inference_messages(mail_text: str, chat_template: str) -> list:
    """
    Build inference messages for the model based on the chat template.

    Parameters
    ----------
    mail_text : str
        The text of the email (ticket description) to be classified.
    chat_template : str
        The template type for chat interaction. Valid options are 'llama' or 'qwen'.

    Returns
    -------
    list of dict
        A list of dictionaries representing the conversation messages in a format
        compatible with the relevant model.
    """
    if chat_template == 'llama':
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_TEMPLATE.format(content=mail_text)
                    }
                ]
            }
        ]
    elif chat_template == 'qwen':
        conversation = [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(content=mail_text)

            }
        ]
    else:
        raise ValueError(f"Unknown chat_template: {chat_template}")
    return conversation


def main():
    """
    Main function to load the model, process the CSV dataset, and generate
    predictions for each row in the dataset. The predictions will be saved
    into a new CSV file.

    Notes
    -----
    - The input CSV file must be named 'test_set_filtred.csv' and must
      contain a column named 'mail'.
    - The output CSV file will contain the original data plus prediction
      columns ('Prediction Type' and 'Prediction').

    Raises
    ------
    ValueError
        If the model type is unknown.
    """

    # Retrieve model configuration
    model_path = MODELS.get(MODEL_NAME)["model_path"]
    model_type = MODELS.get(MODEL_NAME)["model_type"]
    chat_template = MODELS.get(MODEL_NAME)["chat_template"]

    # Load the appropriate model
    if model_type == "instruct":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = True,
            token = HF_TOKEN,
        )
        FastLanguageModel.for_inference(model)

    elif model_type == "multimodal":
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = True,
            token = HF_TOKEN,
        )
        FastVisionModel.for_inference(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Read CSV file
    csv_path = os.path.join(MAIN_PATH, 'test_set_filtred.csv')
    result_df = pd.read_csv(csv_path)

    # If developer testing flag is set, work on a small subset
    if IS_DEV_TEST:
        result_df = result_df.iloc[:20]

    # Prepare for inference
    results = []
    start_time = time.time()

    # Iterate through each row in the dataset
    for index, row in result_df.iterrows():
        if index%50==0:
            print(f'Processing row {index}/{len(result_df)}')

        description = row['mail']
        messages = build_inference_messages(description, chat_template)

        # Convert the message to a text prompt
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False
        )

        if model_type == "instruct":
            inputs = tokenizer(
                prompt_text,
                add_special_tokens = False,
                return_tensors = "pt",
            ).to("cuda")
        else:
            inputs = tokenizer(
                None,  # Used for images in multimodal models
                prompt_text,
                add_special_tokens = False,
                return_tensors = "pt",
            ).to("cuda")

        input_ids = inputs['input_ids']

        # Generate model output
        output = model.generate(
            **inputs,
            max_new_tokens = 50,
            use_cache = True,
            temperature = 1.5,
            min_p = 0.1
        )

        # Extract only the newly generated tokens
        generated_tokens = output[0][len(input_ids[0]):]
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Store result
        results.append({
            "type": "Original Result",
            "categorisation": decoded_output
        })

    end_time = time.time()

    # Merge results with the original DataFrame
    result_df['Prediction Type'] = [result['type'] for result in results]
    result_df['Prediction'] = [result['categorisation'] for result in results]

    # Save results to CSV
    elapsed_seconds = int(end_time - start_time)
    result_file_path = os.path.join(MAIN_PATH, f"{MODEL_NAME}_predictions_duration_{elapsed_seconds}s.csv")
    result_df.to_csv(result_file_path, index=False)

    print(f"Finished processing in {elapsed_seconds} seconds.")
    print(f"Results saved to {result_file_path}")


if __name__ == "__main__":
    main()
