import jieba.posseg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import sys

import gensim
torch.manual_seed(2)
# sys.stdout = open('1.log', 'a')
sent='明天是荣耀运营十周年纪念日。' \
     '荣耀从两周年纪念日开始，' \
     '在每年的纪念日这天凌晨零点会开放一个新区。' \
     '第十版账号卡的销售从三个月前就已经开始。' \
     '在老区玩的不顺心的老玩家、准备进入荣耀的新手，都已经准备好了新区账号对这个日子翘首以盼。' \
    '陈果坐到了叶修旁边的机器，随手登录了她的逐烟霞。' \
     '其他九大区的玩家人气并没有因为第十区的新开而降低多少，' \
     '越老的区越是如此，实在是因为荣耀的一个账号想经营起来并不容易。' \
     '陈果的逐烟霞用了五年时间才在普通玩家中算是翘楚，哪舍得轻易抛弃。' \
     '更何况到最后大家都会冲着十大区的共同地图神之领域去。'
words=jieba.posseg.cut(sent,HMM=True) #分词
processword=[]
tagword=[]
for w in words:
    processword.append(w.word)
    tagword.append(w.flag)
#词语和对应的词性做一一对应
texts=[(processword,tagword)]

#使用gensim构建本例的词汇表
id2word=gensim.corpora.Dictionary([texts[0][0]])
#每个词分配一个独特的ID
word2id=id2word.token2id

#使用gensim构建本例的词性表
id2tag=gensim.corpora.Dictionary([texts[0][1]])
#为每个词性分配ID
tag2id=id2tag.token2id


def sen2id(inputs):
    return [word2id[word] for word in inputs]

def tags2id(inputs):
    return [tag2id[word] for word in inputs]
#根据词汇表把文本输入转换成对应的词汇表的序号张量
def formart_input(inputs):
    return torch.tensor(sen2id(inputs),dtype=torch.long)

#根据词性表把文本标注输入转换成对应的词汇标注的张量
def formart_tag(inputs):
    return torch.tensor(tags2id(inputs),dtype=torch.long)
#定义网络结构
class LSTMTagger(torch.nn.Module):
    def __init__(self,embedding_dim,hidden_dim,voacb_size,target_size):
        super(LSTMTagger,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.voacb_size=voacb_size
        self.target_size=target_size
        # 使用Word2Vec预处理一下输入文本
        self.embedding=nn.Embedding(self.voacb_size,self.embedding_dim)
        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm=nn.LSTM(self.embedding_dim,self.hidden_dim)
        ## 线性层将隐状态空间映射到标注空间
        self.out2tag=nn.Linear(self.hidden_dim,self.target_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self,inputs):

        # 预处理文本转成稠密向量
        embeds=self.embedding((inputs))
        #根据文本的稠密向量训练网络
        out,self.hidden=self.lstm(embeds.view(len(inputs),1,-1),self.hidden)
        #做出预测
        tag_space=self.out2tag(out.view(len(inputs),-1))
        tags=F.log_softmax(tag_space,dim=1)
        return tags


model=LSTMTagger(10,10,len(word2id),len(tag2id))
loss_function=nn.NLLLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)
#看看随机初始化网络的分析结果
with torch.no_grad():
    input_s=formart_input(texts[0][0])
    print(input_s)
    print(processword)
    tag_s=model(input_s)
    for i in range(tag_s.shape[0]):
        print(tag_s[i])
    # print(tag_s)
for epoch in range(300):
    # 再说明下, 实际情况下你不会训练300个周期, 此例中我们只是构造了一些假数据
    for p ,t in texts:
        # Step 1. 请记住 Pytorch 会累加梯度
        # 每次训练前需要清空梯度值
        model.zero_grad()

        # 此外还需要清空 LSTM 的隐状态
        # 将其从上个实例的历史中分离出来
        # 重新初始化隐藏层数据，避免受之前运行代码的干扰,如果不重新初始化，会有报错。
        model.hidden = model.init_hidden()

        # Step 2. 准备网络输入, 将其变为词索引的Tensor 类型数据
        sentence_in=formart_input(p)
        tags_in=formart_tag(t)

        # Step 3. 前向传播
        tag_s=model(sentence_in)

        # Step 4. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss=loss_function(tag_s,tags_in)
        loss.backward()
        print('Loss:',loss.item())
        optimizer.step()

#看看训练后的结果
with torch.no_grad():
    input_s=formart_input(texts[0][0])
    tag_s=model(input_s)
    for i in range(tag_s.shape[0]):
        print(tag_s[i])
