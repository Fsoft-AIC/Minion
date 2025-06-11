<h1 align="left">
  MINION
  <img src="https://raw.githubusercontent.com/pdt0609/Minion/main/png/minion.jpg" width="100" align="right">
</h1>

This is the official implementation of the paper:  
- **Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extraction**

## 🚀 Updates
- \[2025\05\16] Our paper has been accepted to ACL 2025 Main Conference!

## 📊 MINION Framework Visualization

![MINION Model Structure](https://raw.githubusercontent.com/pdt0609/Minion/main/png/model_structure.jpg)




---



## BERT


1. Navigate to the Bert directory:
   ```
   cd Bert
   ```

2. Install necessary packages:
   ```
   !pip install transformers==4.40.0 torch==2.3.0 scikit-learn ==1.4.2 nltk==3.8.1 retry==0.9.2
   !pip install flash-attn --no-build-isolation
   !pip install pytorch_metric_learning
   ```
3. Run the TACRED training:
   ```
   python train_tacred_final.py --task_name Tacred --num_k 5 
   ```
4. Run the FewRel training:
   ```
   python train_fewrel_final.py --task_name FewRel --num_k 5 
   ```
---

## LLM2Vec

### Setup

1. Navigate to the `Llama2`,`Llama3`,`Mistral` in 'LLMEs'directory:

2. Install necessary packages:
   ```
   !pip install transformers==4.40.0 torch==2.3.0 scikit-learn ==1.4.2 nltk==3.8.1 retry==0.9.2
   !pip install llm2vec==0.2.2
   !pip install flash-attn --no-build-isolation
   !pip install pytorch_metric_learning
   ```
3. Log in to Hugging Face:
   ```
   !huggingface-cli login --token your_huggingface_token_to_access_model
   ```

### Running 

1. Run the TACRED training:
   ```
   python train_tacred_final.py --task_name Tacred --num_k 5 
   ```
2. Run the FewRel training:
   ```
   python train_fewrel_final.py --task_name FewRel --num_k 5 
   ```

### Notes

- CPL results with **Llama2** and **Mistral** are evaluated under `float32` for fair comparison with [Quyen et al](https://arxiv.org/abs/2410.00334).
- All other experiments use `bf16`.
- Requires **Python ≥3.8**.
- For CPL, add your OpenAI API key to `config.ini`.
--- 
