The result of CPL using llama2 and mistral are running under float32 for fair comparrision with 


---
!pip install transformers==4.40.0 torch==2.3.0 scikit-learn ==1.4.2 nltk==3.8.1 retry==0.9.2

!pip install pytorch_metric_learning


## BERT


1. Navigate to the Bert directory:
   ```
   cd Bert
   ```
2. Run the TACRED training:
   ```
   python train_tacred_final.py --task_name Tacred --num_k 5 
   ```
3. Run the FewRel training:
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
The result of CPL using llama2 and mistral are running under float32 for fair comparrision with https://arxiv.org/abs/2410.00334. All of the others setting are run under bf16.
For running CPL model put your openai key into config.ini
Requires: Python >=3.8
--- 
