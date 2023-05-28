import torch
from transformers import BertModel, BertTokenizer


class Bert_Embedding_Generator:
    def __init__(self, output_hidden_states=False):
        self.tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=output_hidden_states).to(self.device).eval()

    def encode(self, l):
        #return self.tokenizer.encode(l)
        return self.tokenizer(l, padding=True)
    
    def decode(self, l):
        return self.tokenizer.convert_ids_to_tokens(l)
    
    def __call__(self, sentence, strategy='CLS'):
        enc = self.encode(sentence)
        enc = {k:torch.LongTensor(v).to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        hidden_states = out['last_hidden_state']

        if strategy == 'CLS':
            sentence_embedding = hidden_states[:,0]

        elif strategy == 'average':
            sentence_embedding = torch.mean(hidden_states, dim=1)
        
        if len(sentence) == 1:
            return sentence_embedding.squeeze(0)
        else:
            return sentence_embedding


if __name__ == '__main__':

    s1 = 'i gatti sono animali'
    s2 = 'i cani sono animali'
    s3 = 'cats are animals'
    s4 = 'l\'acciaio non è un essere vivente'
    l = [s1,s2,s3,s4]
    bert = Bert_Embedding_Generator()
    emb_matrix = bert(l)


    cosi = torch.nn.CosineSimilarity(dim=0)
    distance = cosi(emb_matrix[0],emb_matrix[1])
    print(distance)
    print(emb_matrix.shape)
    s1 = 'i gatti sono animali'
    s2 = 'i cani sono animali'
    s3 = 'cats are animals'
    s4 = 'l\'acciaio non è un essere vivente'
    l1 = [s1]
    l2 = [s2]
    l3 = [s3]
    l4 = [s4]
    bert = Bert_Embedding_Generator()

    e1 = bert(l1)
    e2 = bert(l2)
    e3 = bert(l3)
    e4 = bert(l4)

