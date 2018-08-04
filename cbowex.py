import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2  # 左右各2个单词
EMBEDDING_DIM = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# 通过从 `raw_text` 得到一组单词, 进行去重操作
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2*context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

losses = []
loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    total_loss = 0
    for context, target in data:
        # 步骤 1. 准备好进入模型的数据 (例如将单词转换成整数索引,并将其封装在变量中)
        context_idxs = make_context_vector(context, word_to_ix)

        # 步骤 2. 回调 *积累* 梯度. 在进入一个实例前,需要将之前的实力梯度置零
        model.zero_grad()

        # 步骤 3. 运行反向传播,得到单词的概率分布
        log_probs = model(context_idxs)

        # 步骤 4. 计算损失函数. (再次注意, Torch需要将目标单词封装在变量中)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # 步骤 5. 反向传播并更新梯度
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # 在训练集中每次迭代损失都会减小!
# 创建模型并且训练. 这里有一些函数可以在使用模型之前帮助你准备数据




print(data[0])
print(make_context_vector(data[0][0], word_to_ix))   # 例子