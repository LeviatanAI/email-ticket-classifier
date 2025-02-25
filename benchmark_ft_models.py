# -*- coding: utf-8 -*-
"""
Example usage:
--------------
Before running this script, make sure you have installed the required packages:
    !pip install python-Levenshtein

You can then run this script with:
    python prediction_unsloth_ft_models.py

Description:
------------
Improved Script for Prediction Evaluation

This script loads prediction data for different models and platforms,
cleans and normalizes predictions, attempts to match them to a predefined list
of categories, and computes evaluation metrics including accuracy, per-category
precision, recall, and F1-scores, as well as a confusion matrix.

Author:
-------
Leviatan | AI research lab

"""
import os
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from Levenshtein import distance as lev_distance

###############################################################################
# Mutable environment variables
###############################################################################
PREDICTION_NAME = "qwen_ollama_30steps"
#PREDICTION_NAME = "qwen_ollama_1ep"
#PREDICTION_NAME = "qwen_unsloth_30steps"
#PREDICTION_NAME = "qwen_unsloth_1ep"
#PREDICTION_NAME = "llama_unsloth_30steps"
#PREDICTION_NAME = "llama_unsloth_1ep"

# Change the MAIN_PATH variable to match the main path specific to your environment
# The benchmark_ft_models prediction files should be in the MAIN_PATH folder.
MAIN_PATH = "home/ubuntu"


###############################################################################
# Constants
###############################################################################
PREDICTION_FILES = {
    "qwen_ollama_30steps": {
        "file_name": "ollama_prediction_results/Qwen-30steps.csv",
        "plateforme": "ollama"
    },
    "qwen_ollama_1ep": {
        "file_name": "ollama_prediction_results/Qwen-1ep.csv",
        "plateforme": "ollama"
    },
    "qwen_unsloth_30steps": {
        "file_name": "unsloth_prediction_results/Qwen-30steps.csv",
        "plateforme": "unsloth"
    },
    "qwen_unsloth_1ep": {
        "file_name": "unsloth_prediction_results/Qwen-1ep.csv",
        "plateforme": "unsloth"
    },
    "llama_unsloth_30steps": {
        "file_name": "unsloth_prediction_results/Llama-30steps.csv",
        "plateforme": "unsloth"
    },
    "llama_unsloth_1ep": {
        "file_name": "unsloth_prediction_results/Llama-1ep.csv",
        "plateforme": "unsloth"
    },
}


###############################################################################
# Data Loading
###############################################################################

# Retrieve file path and platform used based on PREDICTION_NAME
file_path = PREDICTION_FILES.get(PREDICTION_NAME)["file_name"]
plateforme_used = PREDICTION_FILES.get(PREDICTION_NAME)["plateforme"]

# Load prediction DataFrame
prediction_df = pd.read_csv(os.path.join(MAIN_PATH, file_path))

###############################################################################
# Categories
###############################################################################

# Predefined list of categories for matching
categories = [
    'Demande de service/Backup #BCS/Autre',
    'Demande de service/Backup #BCS/Demande de renseignement',
    'Demande de service/Backup #BCS/Restauration qualifiée',
    'Demande de service/Backup #BCS/Stratégie de sauvegarde/Création',
    'Demande de service/Backup #BCS/Stratégie de sauvegarde/Modification',
    'Demande de service/Backup #BCS/Stratégie de sauvegarde/Suppression',
    "Demande de service/Cyber Sécurité #CS2/Bastion/Création-Modification d'entrées",
    'Incidents/Backup #BCS/Sauvegarde',
    'Incidents/Supervision'
]

# Normalize categories: lowercase, remove extra spaces
categories = [
    re.sub(r'\s+', ' ', cat.lower().strip()) 
    for cat in categories
]

###############################################################################
# Utility Functions
###############################################################################

def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase, removing extra whitespace,
    and optionally removing special characters if needed.

    Parameters
    ----------
    text : str
        The text to normalize.

    Returns
    -------
    str
        The normalized text.
    """
    # Convert to lowercase
    text = text.lower()
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    #text = re.sub(r'[^a-z0-9\s/#-]', '', text)
    return text.strip()


def compute_similarity_scores(text1: str, text2: str) -> Dict[str, float]:
    """
    Compute multiple similarity metrics between two strings.

    Metrics included:
    - Levenshtein distance
    - Absolute difference in lengths
    - SequenceMatcher ratio (a form of substring matching ratio)
    - Word match ratio (intersection of unique words over maximum set size)

    Parameters
    ----------
    text1 : str
        The first text for comparison.
    text2 : str
        The second text for comparison.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the computed similarity metrics.
    """
    # Normalize texts
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    
    # Compute different similarity metrics
    return {
        'levenshtein': lev_distance(norm_text1, norm_text2),
        'length_diff': abs(len(norm_text1) - len(norm_text2)),
        'sequence_ratio': SequenceMatcher(None, norm_text1, norm_text2).ratio(),
        'word_match_ratio': len(set(norm_text1.split()) & set(norm_text2.split())) / max(len(norm_text1.split()), len(norm_text2.split()))
    }


def clean_prediction(text: str) -> str:
    """
    Extract a predicted category from text by looking for content between asterisks.

    The function first searches for content between single asterisks
    that is not preceded or followed by another asterisk.
    If nothing is found, it tries a more lenient pattern.

    Parameters
    ----------
    text : str
        The raw text from which to extract the prediction.

    Returns
    -------
    str
        The cleaned and normalized category text if found,
        otherwise the original text in lowercase.
    """
    # Precise pattern: single asterisks not adjacent to other asterisks
    precise_pattern = r'(?<!\*)\*([^*]+)\*(?!\*)'
    matches = re.findall(precise_pattern, text)
    
    if matches:
        # Return the first match that's not "unknown"
        for match in matches:
            if match.strip().lower() != "unknown":
                return match.strip().lower()
    
    # Lenient pattern: any content between one or more asterisks
    lenient_pattern = r'\*+([^*]+)\*+'
    matches = re.findall(lenient_pattern, text)
    
    if matches:
        # Return the first match that's not "unknown"
        for match in matches:
            if match.strip().lower() != "unknown":
                return match.strip().lower()
    
    # If still no matches, return the cleaned original text
    return text.strip().lower()


def find_best_matching_category(
        prediction: str, categories: List[str],
        lev_threshold: int = 3,
        sequence_ratio_threshold: float = 0.85,
        word_match_threshold: float = 0.8
) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    """
    Find the best matching category for a predicted text among a list of categories.
    Uses multiple similarity metrics (Levenshtein distance, sequence ratio, etc.)
    to identify the category.

    Parameters
    ----------
    prediction : str
        The cleaned prediction text to match.
    categories : List[str]
        List of possible category strings to match against.
    lev_threshold : int, optional
        Maximum allowed Levenshtein distance for a match, by default 10
    sequence_ratio_threshold : float, optional
        Minimum required SequenceMatcher ratio, by default 0.85
    word_match_threshold : float, optional
        Minimum required word match ratio, by default 0.8

    Returns
    -------
    Tuple[Optional[str], Optional[Dict[str, float]]]
        A tuple containing:
        - The best matching category string if found, otherwise None.
        - A dictionary of similarity scores for the match, or None.
    """
    prediction = prediction.lower()
    best_match = None
    best_scores = None
    min_lev_distance = float('inf')

    # Check for an exact match first
    if prediction in categories:
        return categories[categories.index(prediction)], {'exact_match': True}

    # Otherwise, compute similarity metrics with each category
    for i, category in enumerate(categories):
        scores = compute_similarity_scores(prediction, category)
        
        # Check if this is a good match based on multiple criteria
        is_good_match = (
            scores['levenshtein'] <= lev_threshold and
            scores['sequence_ratio'] >= sequence_ratio_threshold and
            scores['word_match_ratio'] >= word_match_threshold
        )
        
        # Update best match if current one is better
        if is_good_match and scores['levenshtein'] < min_lev_distance:
            min_lev_distance = scores['levenshtein']
            best_match = categories[i]
            best_scores = scores
    
    return best_match, best_scores if best_match else None


def evaluate_predictions(
        predictions: pd.DataFrame,
        categories: List[str] = categories,
        lev_threshold: int = 3
) -> Dict:
    """
    Evaluate the model's predictions in a DataFrame against a ground truth column.

    The function cleans the predictions, attempts to match them with the provided
    categories, and then calculates statistics such as overall accuracy, a confusion
    matrix, and per-category precision, recall, and F1.

    Parameters
    ----------
    predictions : pd.DataFrame
        A DataFrame containing at least two columns:
        'Prediction' for model predictions and 'Catégorie du ticket' for ground truth.
    categories : List[str], optional
        List of valid categories to match against, by default CATEGORIES.
    lev_threshold : int, optional
        Maximum allowed Levenshtein distance for a match, by default 10.

    Returns
    -------
    Dict[str, object]
        A dictionary containing:
        - 'total_samples': int, total number of samples
        - 'matched_predictions': int, number of matched predictions
        - 'unmatched_predictions': int, number of unmatched predictions
        - 'matching_rate': float, ratio of matched predictions
        - 'accuracy': float, ratio of correct predictions
        - 'confusion_matrix': pd.DataFrame, confusion matrix with margins
        - 'category_metrics': dict, containing precision, recall, F1, and support
          for each category
        - 'cleaned_data': pd.DataFrame, the input DataFrame with additional columns
          for cleaned predictions and matched predictions
    """
    
    # Clean predictions
    predictions['cleaned_prediction'] = predictions['Prediction'].apply(clean_prediction)
    
    # Find best matching categories
    matches_and_scores = [
        find_best_matching_category(x, categories, lev_threshold) 
        for x in predictions['cleaned_prediction']
    ]
    predictions['matched_prediction'] = [match[0] for match in matches_and_scores]
    predictions['matching_scores'] = [match[1] for match in matches_and_scores]

    total = len(predictions)
    matched = predictions['matched_prediction'].notna().sum()
    unmatched = total - matched

    # Normalize the ground-truth categories
    predictions['Catégorie du ticket'] = predictions['Catégorie du ticket'].apply(normalize_text)

    # Compute accuracy
    correct_predictions = sum(
        predictions['matched_prediction'].fillna('') == predictions['Catégorie du ticket']
    )
    accuracy = correct_predictions / total

    # Confusion matrix
    confusion_matrix = pd.crosstab(
        predictions['Catégorie du ticket'],
        predictions['matched_prediction'].fillna('UNMATCHED'),
        margins=True
    )

    # Per-category metrics
    category_metrics = {}
    for category in categories:
        true_positives = sum(
            (predictions['Catégorie du ticket'] == category) & (predictions['matched_prediction'] == category)
        )
        false_positives = sum(
            (predictions['Catégorie du ticket'] != category) & (predictions['matched_prediction'] == category)
        )
        false_negatives = sum(
            (predictions['Catégorie du ticket'] == category) & (predictions['matched_prediction'] != category)
        )
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        category_metrics[category] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(predictions['Catégorie du ticket'] == category)
        }
    
    return {
        'total_samples': total,
        'matched_predictions': matched,
        'unmatched_predictions': unmatched,
        'matching_rate': matched / total,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'category_metrics': category_metrics,
        'cleaned_data': predictions
    }


###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    if plateforme_used == "ollama":
        # Extract the categorisation value from "Original prediction" column for the "ollama" platform.
        # The regex looks for a pattern like 'categorisation=<CategoryEnum..:"value">' and extracts "value".
        prediction_df['Prediction'] = prediction_df['Original prediction'].apply(
            lambda x: re.search(r"categorisation=<CategoryEnum\..*?:(?:\s*[\"'])(.+?)(?:[\"'])>", x).group(1)
            if re.search(r"categorisation=<CategoryEnum\..*?:(?:\s*[\"'])(.+?)(?:[\"'])>", x)  else x
        )
    elif plateforme_used == "unsloth":
        # Extract the categorisation value from the "Prediction" column for the "unsloth" platform.
        # The regex looks for a pattern like '"categorisation": "value"' and extracts "value".
        prediction_df['Prediction'] = prediction_df['Prediction'].apply(
            lambda x: re.search(r'"categorisation":\s*"(.*?)"', x).group(1)
            if re.search(r'"categorisation":\s*"(.*?)"', x) else "Unknown"
        )
    else:
        ValueError("Unknown platform specified")

    prediction_df = evaluate_predictions(prediction_df)

    # The following print statement provides a summary of the evaluation results.
    # In addition to matched_predictions, unmatched_predictions, matching_rate, and accuracy,
    # you can also print other details from the returned evaluation results, such as:
    # - 'confusion_matrix': A DataFrame showing the confusion matrix with margins.
    # - 'category_metrics': A dictionary containing precision, recall, F1, and support for each category.
    # - 'cleaned_data': The input DataFrame with additional columns for cleaned and matched predictions.
    # To print these, access the corresponding keys in the result dictionary.
    print(f"matched_predictions:{prediction_df['matched_predictions']},"
          f" unmatched_predictions:{prediction_df['unmatched_predictions']}, "
          f"matching_rate:{prediction_df['matching_rate']}, "
          f"accuracy:{prediction_df['accuracy']}")
