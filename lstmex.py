# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# torch.manual_seed(1)
#
# lstm = nn.LSTM(3, 3)  # 输入维度是3, 输出维度也是3
# # print('LSTM的所有权重',lstm.all_weights)
#
# inputs = [torch.randn(1, 3) for _ in range(5)] # 构造一个长度为5的序列
#
# print('Inputs:',inputs)
#
# # 初始化隐藏状态
# hidden = (torch.randn(1, 1, 3),
#           torch.randn(1, 1, 3))
# print('Hidden:',hidden)
#
# for i in inputs:
#     # 将序列的元素逐个输入到LSTM
#     # 经过每步操作,hidden 的值包含了隐藏状态的信息
#     out, hidden = lstm(i.view(1, 1, -1), hidden)
# print('out1:',out)
# print('hidden2:',hidden)
# # 另外, 我们还可以一次对整个序列进行训练. LSTM 返回的第一个值表示所有时刻的隐状态值,
# # 第二个值表示最近的隐状态值 (因此下面的 "out"的最后一个值和 "hidden" 的值是一样的).
# # 之所以这样设计, 是为了通过 "out" 的值来获取所有的隐状态值, 而用 "hidden" 的值来
# # 进行序列的反向传播运算, 具体方式就是将它作为参数传入后面的 LSTM 网络.
#
# # 增加额外的第二个维度
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)
# print('out2',out)
# print('hidden3',hidden)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gensim
torch.manual_seed(2)

datas=[('你 叫 什么 名字 ?','n v n n f'),
       ('今天 天气 怎么样 ?','n n adj f'),]
words=[ data[0].split() for data in datas]
tags=[ data[1].split() for data in datas]
print(words)
print(tags)

id2word=gensim.corpora.Dictionary(words)
word2id=id2word.token2id
# print('id2word',id2word)
# print('word2id',word2id)

id2tag=gensim.corpora.Dictionary(tags)
tag2id=id2tag.token2id
print('id2tag',id2tag)
print('tag2id',tag2id)

#句子转成成
def sen2id(inputs):
    return [word2id[word] for word in inputs]
def tags2id(inputs):
    return [tag2id[word] for word in inputs]
# print(sen2id('你 叫 什么 名字'.split()))

def formart_input(inputs):
    return torch.tensor(sen2id(inputs),dtype=torch.long)
def formart_tag(inputs):
    return torch.tensor(tags2id(inputs),dtype=torch.long)

class LSTMTagger(torch.nn.Module):
    def __init__(self,embedding_dim,hidden_dim,voacb_size,target_size):
        super(LSTMTagger,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.voacb_size=voacb_size
        self.target_size=target_size
        self.embedding=nn.Embedding(self.voacb_size,self.embedding_dim)
        self.lstm=nn.LSTM(self.embedding_dim,self.hidden_dim)
        self.out2tag=nn.Linear(self.hidden_dim,self.target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self,inputs):
        embeds=self.embedding((inputs))
        out,self.hidden=self.lstm(embeds.view(len(inputs),1,-1),self.hidden)
        tag_space=self.out2tag(out.view(len(inputs),-1))
        tags=F.log_softmax(tag_space,dim=1)
        return tags


model=LSTMTagger(6,6,len(word2id),len(tag2id))
loss_function=nn.NLLLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)
with torch.no_grad():
    input_s=formart_input(datas[0][0].split())
    print(datas[0])
    tag_s=model(input_s)
    print(tag_s)
for epoch in range(300):
    # print('epoch:',epoch)
    for sentenct,tags in datas:
        model.zero_grad()
        # 重新初始化隐藏层数据，避免受之前运行代码的干扰,如果不重新初始化，会有报错。
        model.hidden = model.init_hidden()
        sentence_in=formart_input(sentenct.split())
        # print('sentence_in:',sentence_in)
        tags_in=formart_tag(tags.split())
        # print('tags_in:',tags_in)
        tag_s=model(sentence_in)
        loss=loss_function(tag_s,tags_in)
        loss.backward()
        # print('Loss:',loss.item())
        optimizer.step()

with torch.no_grad():
    input_s=formart_input(datas[0][0].split())
    print(datas[0])
    tag_s=model(input_s)
    print(tag_s)