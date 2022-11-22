from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import numpy as np
from PIL import Image

class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer=None, tokenizer_out_dim=64, type='title'):
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer_out_dim = tokenizer_out_dim
        self.type = type

        # encode labels
        labels = sorted(set([t['genre'] for t in data]))
        self.id2label = {k: v for k, v in enumerate(labels)}
        self.label2id = {v: k for k, v in enumerate(labels)}

    def __getitem__(self, index):

        if self.tokenizer is None:
            raise SystemError('Tokenizer cannot be None. Please set tokenizer by calling set_tokenizer(tokenizer) function.')
        
        labels = self.label2id[self.data[index]['genre']]

        # tokenize and load image as array
        # assign every variable with corresponding label
        if self.type == 'title':
            title = self.tokenize(self.data[index]['title'])
            title['labels'] = labels
            return title
        elif self.type == 'description':
            desc = self.tokenize(self.data[index]['simple_desc'])
            desc['labels'] = labels
            return desc
        elif self.type == 'image':
            img = self.load_img(self.data[index]['img_local_path'], self.data[index]['title'])
            img['labels'] = labels            
            return img
        else:
            raise NameError()


    def __len__(self):
        return len(self.data)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def load_img(self, url, title=None):
        try:
            # feature = read_image(url)
            pil_img = Image.open(url).convert('RGB')
        except:            
            # make url adjustment
            template = '\/:*?&”;%<>\'|'            
            ntitle = title.translate(str.maketrans(template, '_' * len(template), ''))            
            pil_img = Image.open(url.replace(title, ntitle)).convert('RGB')

        
        pil_feature = self.tokenizer(pil_img)['pixel_values'][0]
                    
        return {
                'pixel_values': pil_feature,
                'image': np.array(pil_img)
            }

    def tokenize(self, item):

        item = item.strip()
        
        encoded = self.tokenizer(
            item,
            add_special_tokens=True,
            max_length=self.tokenizer_out_dim,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        encoded['input_ids'] = encoded['input_ids'].squeeze(0)
        encoded['token_type_ids'] = encoded['token_type_ids'].squeeze(0)
        encoded['attention_mask'] = encoded['attention_mask'].squeeze(0)

        return encoded