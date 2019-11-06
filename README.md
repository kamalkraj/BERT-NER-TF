# BERT NER

Use google BERT to do CoNLL-2003 NER !

Train model using Python and TensorFlow 2.0

[ALBERT-TF2.0](https://github.com/kamalkraj/ALBERT-TF2.0)

[BERT-SQuAD](https://github.com/kamalkraj/BERT-SQuAD)

[BERT-NER-Pytorch](https://github.com/kamalkraj/BERT-NER)


# Requirements

- `python3`
- `pip3 install -r requirements.txt`

### Download Pretrained Models from Tensorflow offical models
- [bert-base-cased](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/cased_L-12_H-768_A-12.tar.gz) unzip into `bert-base-cased`

- [bert-large-cased](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/tf_20/cased_L-24_H-1024_A-16.tar.gz) unzip into `bert-large-cased`

code for pre-trained bert from [tensorflow-offical-models](https://github.com/tensorflow/models/tree/master/official/nlp) 

# Run

## Single GPU

`python run_ner.py --data_dir=data/ --bert_model=bert-base-cased --output_dir=out_base --max_seq_length=128 --do_train --num_train_epochs 3 --do_eval --eval_on dev`

## Multi GPU

`python run_ner.py --data_dir=data/ --bert_model=bert-large-cased --output_dir=out_large --max_seq_length=128 --do_train --num_train_epochs 3 --multi_gpu --gpus 0,1,2,3 --do_eval --eval_on test`

# Result

## BERT-BASE

### Validation Data
```
             precision    recall  f1-score   support

        PER     0.9677    0.9756    0.9716      1842
        LOC     0.9671    0.9592    0.9631      1837
       MISC     0.8872    0.9132    0.9001       922
        ORG     0.9191    0.9314    0.9252      1341

avg / total     0.9440    0.9509    0.9474      5942
```
### Test Data
```
             precision    recall  f1-score   support

        ORG     0.8773    0.9037    0.8903      1661
        PER     0.9646    0.9592    0.9619      1617
       MISC     0.7691    0.8305    0.7986       702
        LOC     0.9333    0.9305    0.9319      1668

avg / total     0.9053    0.9184    0.9117      5648
```
## Pretrained model download from [here](https://drive.google.com/file/d/1ZlQimY5xbkpS_1baO-ZtCZZef4MvG9__/view?usp=sharing)

## BERT-LARGE

### Validation Data
```
             precision    recall  f1-score   support

        ORG     0.9290    0.9374    0.9332      1341
       MISC     0.8967    0.9230    0.9097       922
        PER     0.9713    0.9734    0.9723      1842
        LOC     0.9748    0.9701    0.9724      1837

avg / total     0.9513    0.9564    0.9538      5942
```
### Test Data
```
             precision    recall  f1-score   support

        LOC     0.9256    0.9329    0.9292      1668
       MISC     0.7891    0.8419    0.8146       702
        PER     0.9647    0.9623    0.9635      1617
        ORG     0.8903    0.9133    0.9016      1661

avg / total     0.9094    0.9242    0.9167      5648
```
## Pretrained model download from [here](https://drive.google.com/file/d/1BZCKj_e_SXxlvg4rKUC0EI4BJXYssTVL/view?usp=sharing)

# Inference

```python
from bert import Ner

model = Ner("out_base/")

output = model.predict("Steve went to Paris")

print(output)
'''
    [
        {
            "confidence": 0.9981840252876282,
            "tag": "B-PER",
            "word": "Steve"
        },
        {
            "confidence": 0.9998939037322998,
            "tag": "O",
            "word": "went"
        },
        {
            "confidence": 0.999891996383667,
            "tag": "O",
            "word": "to"
        },
        {
            "confidence": 0.9991968274116516,
            "tag": "B-LOC",
            "word": "Paris"
        }
    ]
'''
```

# Deploy REST-API
BERT NER model deployed as rest api
```bash
python api.py
```
API will be live at `0.0.0.0:8000` endpoint `predict`
#### cURL request
` curl -X POST http://0.0.0.0:8000/predict -H 'Content-Type: application/json' -d '{ "text": "Steve went to Paris" }'`

Output
```json
{
    "result": [
        {
            "confidence": 0.9981840252876282,
            "tag": "B-PER",
            "word": "Steve"
        },
        {
            "confidence": 0.9998939037322998,
            "tag": "O",
            "word": "went"
        },
        {
            "confidence": 0.999891996383667,
            "tag": "O",
            "word": "to"
        },
        {
            "confidence": 0.9991968274116516,
            "tag": "B-LOC",
            "word": "Paris"
        }
    ]
}
```
#### cURL 
![curl output image](/img/curl.png)
#### Postman
![postman output image](/img/postman.png)


### Pytorch version

- https://github.com/kamalkraj/BERT-NER
