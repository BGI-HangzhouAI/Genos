# Text-genome model fusion
## 1. Project Overview
This project is a DNA sequence analysis system based on DNA model + text model, primarily used for gene variant effect prediction and disease association analysis. The project combines DNA sequence encoders and large language models to achieve deep understanding and analysis of DNA sequences.

## 2. Data Analysis
### 2.1 Input Data Format
#### KEGG Dataset
● File Format: JSON format

● Data Structure:
  ```json
  {
    "question": "Problem description defined by chromosome information and pathway networks",
    "answer": "Disease name (e.g., cushing syndrome, parkinson's disease, amyotrophic lateral sclerosis)",
    "reasoning": "Detailed reasoning steps, including 10 steps of biological analysis",
    "reference_sequence": "Reference DNA sequence (uppercase letters, spaces removed)",
    "variant_sequence": "Variant DNA sequence (uppercase letters, spaces removed)"
  }
  ```
  ### 2.2 Output Data Format
#### 2.2.1 Model Output
● Text Generation: Model generates text containing reasoning process and final answer

● Format:
  ```
  <|im_start|>assistant
  [Reasoning content]
  Answer: [Final answer]<|im_end|>
  ```

#### 2.2.2 Evaluation Results
● CSV Format: Comparison of prediction results and ground truth labels

● Column Structure:

    ○ ground_truth: Ground truth label
    ○ pred_label: Predicted label
    ○ generated_text: Complete generated text

## 3. Data Preprocessing Pipeline
### 3.1 DNA Sequence Preprocessing
#### 3.1.1 Sequence Normalization
● Case Conversion: All DNA sequences converted to uppercase letters

● Space Removal: Remove whitespace characters from sequences

● Sequence Truncation: Use ```truncate_dna``` function to truncate sequences from both ends
 ```python
  def truncate_dna(example, truncate_dna_per_side=1024):
      # Truncate 1024 base pairs from each end of the sequence
      # If sequence is too short, return the middle portion
  ```

#### 3.1.2 Sequence Tokenization
● OURGEN Tokenizer: Use character-level tokenizer to process DNA sequences

● Special Tokens: Add <|dna_start|>, <|dna_pad|>, <|dna_end|> tokens

● Sequence Length Limit: Maximum length of 2048 tokens
### 3.2 Text Preprocessing
#### 3.2.1 Dialogue Format Conversion
● Multimodal Format: Dialogue format combining DNA sequences and text

● Role Definition:

    ○ user: Contains DNA sequences and questions
    ○ assistant: Contains reasoning process and answers
#### 3.2.2 Template Application
● Chat Template: Use custom chat template to format input

● Special Token Handling: Properly handle ```<|im_start|>``` and ```<|im_end|>``` tokens

### 3.3 Data Loading and Batching
#### 3.3.1 Dataset Split
● Training Set: 80%

● Validation Set: 10%

● Test Set: 10%
#### 3.3.2 Batching Function
● qwen_dna_collate_fn: Batching function specifically designed for Qwen DNA model

● Label Masking: Only compute loss for assistant response part

● Padding Handling: Left padding strategy, add special tokens

## 4. Evaluation Metrics
### 4.1 Classification Metrics
#### 4.1.1 Basic Metrics
● Accuracy: Proportion of correctly predicted samples out of total samples

● Precision: Macro-average precision, average of precision across all classes

● Recall: Macro-average recall, average of recall across all classes

● F1-Score: Macro-average F1 score, harmonic mean of precision and recall
#### 4.1.2 Calculation Method
```python
# Use sklearn's classification_report
report_dict = classification_report(
    y_true, y_pred, 
    labels=labels, 
    output_dict=True, 
    zero_division=1
)

# Extract macro-average metrics
macro_metrics = report_dict['macro avg']
Accuracy = accuracy_score(ground_truth, pred_label)
Precision = macro_metrics['precision']
Recall = macro_metrics['recall'] 
F1_score = macro_metrics['f1-score']
```
### 4.2 Evaluation Process
#### 4.2.1 Validation Phase Evaluation
● Real-time Generation: Generate text on validation set

● Answer Extraction: Extract answer portion from generated text

● Metric Calculation: Calculate classification metrics and log them
#### 4.2.2 Test Phase Evaluation
● Complete Evaluation: Perform complete evaluation on test set

● Result Saving: Save prediction results as CSV file

● Metric Summary: Generate detailed classification report
### 4.3 Special Handling
#### 4.3.1 Answer Extraction
● Regular Matching: Use ```extract_single_entry``` function to extract answers

● Format Processing: Handle prediction results containing ```<think>``` prefix

● Error Handling: Return NaN for answers that cannot be extracted
#### 4.3.2 Multi-class Support
● Dynamic Labels: Dynamically determine classes based on ground truth labels

● Zero Division Handling: Use zero_division=1 to handle cases with no predictions

● Macro-average: Use macro-average to ensure all classes are treated equally

## 5. Model Architecture
### 5.1 Core Components
● DNA Encoder: Evo2 model for DNA sequence encoding

● Text Model: Qwen for text understanding and generation

● Projection Layer: Project DNA features to text embedding space

● LoRA Adaptation: Use LoRA for parameter-efficient fine-tuning
### 5.2 Training Strategy
● Freezing Strategy: Optional freezing of DNA encoder or text model

● Mixed Precision: Use DeepSpeed strategy for efficient training

● Gradient Accumulation: Support large batch training

● Mixed Precision: Support large batch training


## 6. Usage
### 6.1 Environment Setup
This project requires running on a machine with GPU. Please check if CUDA is installed on your machine first.
```python
# Check environment
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```
The project provides a requirements.txt file. Users can install dependencies using the following command:
```shell
pip install -r requirements.txt
```
### 6.2 Model Training
This project provides two running methods: shell script and jupyter notebook.
#### 6.2.1 Shell Script
Users can start model training by running `sh_train.sh`.
#### 6.2.2 Notebook
Users can also start model training by running the notebook file `user_case.ipynb`.
### 6.3 Model Deployment
This project provides a FastAPI-based approach for model deployment. Users can start the service by running the script `start_simple.sh`, noting to modify the model parameters. Then test the service by running the script `test_simple_api.py`.

