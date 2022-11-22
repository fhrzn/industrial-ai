from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import AutoFeatureExtractor, ResNetForImageClassification

def get_bert_model(checkpoint, num_labels=24):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

    return tokenizer, model

def get_resnet_model(checkpoint, num_labels=24):
    tokenizer = AutoFeatureExtractor.from_pretrained(checkpoint)
    model = ResNetForImageClassification.from_pretrained(checkpoint, num_labels=num_labels)

    return tokenizer, model

def get_vit_model(checkpoint, num_labels=24):
    tokenizer = ViTFeatureExtractor.from_pretrained(checkpoint)
    model = ViTForImageClassification.from_pretrained(checkpoint, num_labels=num_labels)

    return tokenizer, model
