import pandas as pd

def create_label_from_meta_csv(meta_csv: pd.DataFrame, label_key: str):
    """
    Create a label DataFrame from a meta CSV file.

    Args:
        meta_csv (str): Path to the meta CSV file.
        label_key (str): Key to use for labeling.

    Returns:
        pd.DataFrame: DataFrame containing labels.
    """
    if label_key == "case_control_label":
        labels  = meta_csv[label_key].apply(lambda x: 1 if x == "CASE" else 0).values
    elif label_key == "immediate_responder" or label_key == "meaningful_responder":
        labels  = meta_csv[label_key].apply(lambda x: 1 if x == "Responder" else 0).values
    else:
        raise ValueError(f"Unknown label key: {label_key}")
    return labels
