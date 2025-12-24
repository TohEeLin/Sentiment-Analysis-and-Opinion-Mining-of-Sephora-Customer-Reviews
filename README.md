# Sentiment Analysis of Sephora Reviews

This project presents an end-to-end NLP pipeline for analyzing customer sentiment from Sephora product reviews. The project is divided into two main parts:
1.  **Sentiment Classification:** Evaluating various machine learning and deep learning models to classify overall review sentiment.
2.  **Opinion Mining & Visualization:** Performing Aspect-Based Sentiment Analysis (ABSA) to extract detailed insights about specific product features and visualizing these findings.

## Dataset

- **Source**: Sephora product review dataset (https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)
- **Note on Provided Data:** The initial dataset consists of multiple large files. Due to GitHub's file size limits, only the final, processed dataset used for modeling (**`full_sephora_reviews_classification.csv`**) is included in this repository. This dataset has been merged, cleaned and balanced via downsampling (2600 reviews per class) to ensure fair model evaluation.  
- **Classes**: Positive, Neutral, Negative  
- **Split**: 60% Training, 20% Validation, 20% Testing

---

## Part 1: Sentiment Classification

This project explores the effectiveness of various machine learning and deep learning models in classifying customer sentiment from Sephora product reviews. The models include Logistic Regression, SVM, Pre-Trained BERT, CNN, and Fine-Tuned BERT. The goal is to evaluate how well each model can detect **positive**, **neutral**, and **negative** sentiments from user-generated text.

### Models Used

- **Logistic Regression (TF-IDF)**
- **Support Vector Machine (TF-IDF)**
- **BERT as Feature Extractor (No Fine-Tuning)**
- **Convolutional Neural Network (Trainable Embeddings)**
- **Fine-Tuned BERT (Transformer-based Classification)**

Each model was evaluated using **accuracy**, **precision**, **recall**, and **F1-score**, along with **confusion matrix** analysis.

### Results Summary

| Model                  | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 67.69%   | 67.42%    | 67.69% | 67.53%   |
| SVM                   | 65.32%   | 64.92%    | 65.32% | 65.07%   |
| BERT (Feature Extractor) | 69.68% | 70.54% | 69.68% | 69.93%   |
| CNN                   | 71.00%   | 70.00%    | 71.00% | 70.00%   |
| Fine-Tuned BERT       | **79.00%** | **79.00%** | **79.00%** | **79.00%** |

### Key Findings

- Fine-Tuned BERT outperformed all other models across all metrics.
- CNN and Pre-Trained BERT also showed strong performance, surpassing traditional models.
- Neutral reviews were generally the most difficult to classify correctly, especially for non-contextual models.

---

## Part 2: Visualization and Insight Generation

Beyond classification, a **rule-based ABSA system** was developed to extract customer opinions on specific product aspects. This allows for a deeper understanding of **why** customers feel a certain way.

### Tools & Libraries Used

- **spaCy**
- **VADER**

### Key Visualizations & Insights

Here are some of the key insights derived from the ABSA results.

**1. High-Level Sentiment Language: Positive vs. Negative Word Clouds**

The word clouds provide a visual summary of the language used in reviews. Positive reviews are dominated by affirmative words like "love", "great" and "smooth", focusing on desired effects. Negative reviews are characterized by problem-oriented terms like "breakout", "dry" and "acne", highlighting negative skin responses.

![Screenshot 2025-06-23 220113](https://github.com/user-attachments/assets/8ccd2813-fe29-4b79-ac2a-ad3241c7f8e8)


**2. Most Discussed Product Aspects**

This chart identifies the top 10 topics customers discuss most. "Skin" is the most frequent aspect, followed by general discussion of the "product." Specific product types like "moisturizer" and "cream" as well as sensory attributes like "smell" and "texture" are also key areas of focus.

![Screenshot 2025-06-23 222805](https://github.com/user-attachments/assets/d0775958-d19b-4429-bb94-37c938714dd7)


**3. Sentiment Breakdown by Product Aspect**

This chart reveals the sentiment behind the most discussed topics. It shows that although "skin" is mentioned most often, it has a high volume of negative sentiment, meaning customers are more likely to leave a review when a product causes a bad skin reaction than when it works well. In contrast, sensory aspects like "texture" and "scent" receive mostly positive feedback, highlighting their importance for customer satisfaction.

![Screenshot 2025-06-23 224852](https://github.com/user-attachments/assets/9a3eecb8-7ad7-4d8e-a420-23403317edde)


**4. Most Common Positive and Negative Opinion Words**

These charts identify the exact words that signal satisfaction or complaint. Positive opinions are led by strong words like "love" and "good." Negative opinions are dominated by terms describing product failures such as "dry", "sensitive" and "oily," providing clear, actionable feedback for product improvement.

![Screenshot 2025-06-23 231342](https://github.com/user-attachments/assets/0eead03c-48cb-40cc-8587-9489f2c65eca)


---

## Future Work

Future improvements could include:
- Training on **larger and more diverse datasets** to improve generalization.
- Implementing **hyperparameter tuning** (e.g., grid search, random search) to optimize learning rate, batch size, dropout, and regularization.
- Exploring **transformer variants** like RoBERTa or DistilBERT.
- Using **ensemble methods** for performance boosting.
- Adding **explainability tools** like SHAP or LIME to interpret model decisions, especially in user-facing applications.
