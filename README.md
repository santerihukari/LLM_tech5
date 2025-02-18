Errors and Fixes
# 1. Dependency Conflicts
### Error:

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gensim 4.3.3 requires scipy<1.14.0,>=1.7.0, but you have scipy 1.15.2 which is incompatible.
mlxtend 0.23.3 requires scikit-learn>=1.3.1, but you have scikit-learn 1.2.2 which is incompatible.
plotnine 0.14.4 requires matplotlib>=3.8.0, but you have matplotlib 3.7.5 which is incompatible.
ydata-profiling 4.12.1 requires scipy<1.14,>=1.4.1, but you have scipy 1.15.2 which is incompatible.
### Fix:
Install specific versions of conflicting packages:

!pip install -q -U scikit-learn==1.3.1 matplotlib==3.8.0
!pip install -q -U bitsandbytes transformers peft accelerate datasets scipy==1.13.0 einops evaluate trl rouge_score
# 2. rouge_score Version Error
### Error:

AttributeError: module 'rouge_score' has no attribute '__version__'
### Fix:
Use importlib.metadata to check the version of rouge_score:

from importlib.metadata import version
try:
    rouge_version = version('rouge_score')
    print("Rouge Score version:", rouge_version)
except Exception as e:
    print("Could not determine rouge_score version:", e)
# 3. LoRA Target Modules Error
### Error:

ValueError: Target modules {'q_proj', 'dense', 'v_proj', 'k_proj'} not found in the base model. Please check the target modules and try again.
### Fix:
Update the LoraConfig to use the correct target modules for GPT-2:

config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['c_attn', 'c_proj', 'c_fc'],  # Correct modules for GPT-2
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
# 4. Checkpoint Directory Not Found
### Error:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/peft-dialogue-summary-training/final-checkpoint/checkpoint-1000'
### Fix:
Ensure the checkpoint directory exists and contains the required files (adapter_config.json and adapter_model.bin). If not, re-save the model:

peft_model.save_pretrained("/kaggle/working/peft-dialogue-summary-training/final-checkpoint/checkpoint-500")
# 5. Deprecation Warnings
### Warning:

FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead.
### Fix:
Replace evaluation_strategy with eval_strategy in TrainingArguments:

peft_training_args = TrainingArguments(
    eval_strategy="steps",  # Updated from evaluation_strategy
    ...
)
# 6. ROUGE Score Improvement
After fine-tuning, the PEFT model showed significant improvement over the original model:

Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL:
- rouge1: 14.66%
- rouge2: 4.48%
- rougeL: 10.89%
- rougeLsum: 12.96%
