from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import AutoFeatureExtractor, ResNetForImageClassification

def get_bert_model(checkpoint, label2id=None, id2label=None, num_labels=24, problem_type='multi_label_classification'):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels, 
        problem_type=problem_type, id2label=id2label,
        label2id=label2id)

    return tokenizer, model

def get_resnet_model(checkpoint, label2id=None, id2label=None, num_labels=24, problem_type='multi_label_classification'):
    tokenizer = AutoFeatureExtractor.from_pretrained(checkpoint)
    model = ResNetForImageClassification.from_pretrained(
        checkpoint, num_labels=num_labels, 
        problem_type=problem_type, id2label=id2label,
        label2id=label2id)

    return tokenizer, model

def get_vit_model(checkpoint, label2id=None, id2label=None, num_labels=24, problem_type='multi_label_classification'):
    tokenizer = ViTFeatureExtractor.from_pretrained(checkpoint)
    model = ViTForImageClassification.from_pretrained(
        checkpoint, num_labels=num_labels, 
        problem_type=problem_type, id2label=id2label,
        label2id=label2id)

    return tokenizer, model
