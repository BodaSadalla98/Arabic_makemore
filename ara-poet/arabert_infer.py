import torch
device = 0 if torch.cuda.is_available() else 'cpu'

print('Using device:', device)

from transformers import GPT2TokenizerFast, pipeline
#for base and medium
from transformers import GPT2LMHeadModel
#for large and mega
# pip install arabert
from arabert.aragpt2.grover.modeling_gpt2 import GPT2LMHeadModel

from arabert.preprocess import ArabertPreprocessor

# MODEL_NAME='/data/boda/Arabic_makemore/ara-poet/checkpoint-254919'
MODEL_NAME='aubmindlab/aragpt2-large'

arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME)


model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
generation_pipeline = pipeline("text-generation",model=model,tokenizer=tokenizer, device=device)

text="عيناك غابتا نخيل ساعة السحر او شرفتان راح يناي عنهما القمر عيناك حين"
text_clean = arabert_prep.preprocess(text)


with open('out.txt' ,'w') as f:

    for i in range(5):
        text = generation_pipeline(text_clean)[0]['generated_text']
        f.write(text)
        f.write('\n\n--------------------------------------------------------\n\n')


