'''
T5-base finetuned on 150k medium articles
'''

import streamlit as st

import subprocess
import sys


# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install('transformers')
# install('nltk')
# install('torch')

def clear_text():
    st.session_state['text'] = ''

from transformers import AutoTokenizer
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk

# model_name = "checkpoint-2200"

# model_dir = f"{model_name}"

st.title('T5 Finetuned summarizator :sunglasses:')

st.write('Loading model...')

tokenizer = AutoTokenizer.from_pretrained("user336/t5-sum-checkpoint-2200")
model = AutoModelForSeq2SeqLM.from_pretrained("user336/t5-sum-checkpoint-2200")

st.write('Done!')
# st.text_input("Input text to summarize", key="text")

# You can access the value at any point with:
# st.session_state.text


# st.title('A title with _italics_ :blue[colors] and emojis :sunglasses:')

text = st.text_area("Input text to summarize",
                    help='Best quality of summarization achieved when placing more than one sentence of text B-)'
                    ,key="text")

if text != '':
    st.write('Our AI already reading your input and making a summary. Please wait... ')
    max_input_length = 512

    inputs = ["summarize: " + text]

    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")

    output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

    # print(predicted_title)
    
    st.write('Here is your summary!')
    st.write(predicted_title)
    
    st.button('Clear',on_click=clear_text)
    # st.write('input')


    # st.info('This is a purely informational message. If you click the button below, there will be celebration!')
    # if st.button('Click for celebration'):
    #     st.balloons()
    #     st.balloons()