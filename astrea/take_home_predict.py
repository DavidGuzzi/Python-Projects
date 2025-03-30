import numpy as np
import pandas as pd
import statsmodels.api as sm

def load_model_coefficients(filepath):
    """Loads the fitted model coefficients from a CSV file."""
    coefs_df = pd.read_csv(filepath)
    coefs = coefs_df.set_index("Variable")["Coefficient"].to_dict()
    return coefs

def compute_choice_probabilities(products_df, coefs):
    """Computes choice probabilities using the MNL model."""
    # Ensure a constant term for the intercept
    products_df = products_df.copy()
    products_df["const"] = 1.0
    
    # Compute utility scores
    utilities = (products_df["const"] * coefs.get("const", 0) +
                 products_df["price"] * coefs.get("price", 0) +
                 products_df["brand_strength"] * coefs.get("brand_strength", 0) +
                 products_df["quality_score"] * coefs.get("quality_score", 0))
    
    # Apply softmax transformation
    exp_utilities = np.exp(utilities)
    choice_probs = exp_utilities / exp_utilities.sum()
    
    products_df["predicted_choice_prob"] = choice_probs
    return products_df

# Example usage:
if __name__ == "__main__":
    # Load model coefficients
    coefs = load_model_coefficients("logit_coefficients_updated.csv")
    
    # Define a sample set of products (e.g., from the original dataset)
    sample_products = pd.DataFrame({
        "product_id": [1, 2, 3, 4, 5],
        "price": [5.0, 7.0, 6.5, 8.0, 4.5],
        "brand_strength": [0.3, 0.6, 0.2, 0.8, 0.5],
        "quality_score": [3.0, 4.5, 2.5, 4.0, 3.5]
    })
    
    # Compute predicted choice probabilities
    predictions = compute_choice_probabilities(sample_products, coefs)
    print(predictions[["product_id", "predicted_choice_prob"]])
