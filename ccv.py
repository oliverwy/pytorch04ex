import torch
import gensim
torch.manual_seed(2)

datas=[('你 叫 什么 名字 ?','n v n n f'),('今天 天气 怎么样 ?','n n adj f'),]
words=[ data[0].split() for data in datas]
tags=[ data[1].split() for data in datas]


id2word=gensim.corpora.Dictionary(words)
word2id=id2word.token2id

id2tag=gensim.corpora.Dictionary(tags)
tag2id=id2tag.token2id

def sen2id(inputs):
    return [word2id[word] for word in inputs]
def tags2id(inputs):
    return [tag2id[word] for word in inputs]
# print(sen2id('你 叫 什么 名字'.split()))
def formart_input(inputs):
    return torch.autograd.Variable(torch.LongTensor(sen2id(inputs)))
def formart_tag(inputs):
    return torch.autograd.Variable(torch.LongTensor(tags2id(inputs)),)

class LSTMTagger(torch.nn.Module):
    def __init__(self,embedding_dim,hidden_dim,voacb_size,target_size):
        super(LSTMTagger,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.voacb_size=voacb_size
        self.target_size=target_size
        self.lstm=torch.nn.LSTM(self.embedding_dim,self.hidden_dim)
        self.log_softmax=torch.nn.LogSoftmax()
        self.embedding=torch.nn.Embedding(self.voacb_size,self.embedding_dim)
        self.hidden=(torch.autograd.Variable(torch.zeros(1,1,self.hidden_dim)),torch.autograd.Variable(torch.zeros(1,1,self.hidden_dim)))
        self.out2tag=torch.nn.Linear(self.hidden_dim,self.target_size)
    def forward(self,inputs):
        input=self.embedding((inputs))
        out,self.hidden=self.lstm(input.view(-1,1,self.embedding_dim),self.hidden)
        tags=self.log_softmax(self.out2tag(out.view(-1,self.hidden_dim)))
        return tags

model=LSTMTagger(3,3,len(word2id),len(tag2id))
loss_function=torch.nn.NLLLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
for _ in range(100):
    model.zero_grad()
    input=formart_input('你 叫 什么 名字'.split())
    tags=formart_tag('n n adj f'.split())
    out=model(input)
    loss=loss_function(out,tags)
    loss.backward(retain_variables=True)
    optimizer.step()
    print(loss.data[0])
input=formart_input('你 叫 什么'.split())
out=model(input)
out=torch.max(out,1)[1]
print([id2tag[out.data[i]] for i in range(0,out.size()[0])])