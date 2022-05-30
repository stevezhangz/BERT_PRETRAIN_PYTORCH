# BERT_PRETRAIN_PYTORCH

how to run

    git clone https://github.com/stevezhangz/BERT_PRETRAIN_PYTORCH.git
    cd BERT_PRETRAIN_PYTORCH
    mkdir dataset
    # move txt files to 'dataset' such as dataset/oscar.eo.txt
    pip install -r requirements.txt
    python create_sentence_pairs.py
    python train_tokenizer.py
    python main.py
