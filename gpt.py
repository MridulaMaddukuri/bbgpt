"""
Following along with the tutorial from
https://www.youtube.com/watch?v=kCc8FmEb1nY

This script
1. reads an input txt file
2. Encodes the text
3. Creates a tensor for the input text
4.
"""

import torch
from bigram_model import BigramLanguageModel


# read the file
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# print the first 1000 characters
print(text[:1000])


# get the unique characters in the text
chars = sorted(set(text))
print(chars)
vocab_size = len(chars)
print(vocab_size)


# Character level tokenization: convert raw text to a sequence of integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


print(encode("welcome"))
print(decode(encode("welcome")))

# NOTE:
# This is a very simple imprementation of tokenization. There are several other
# tokenization libraries that are more sophisticated and can handle more complex
# tasks such as subword tokenization, sentencepiece, etc.
# Google uses sentencepiece for their tokenization
# https://github.com/google/sentencepiece
# tiktoken is the tokenizer used by OpenAI
# https://github.com/openai/tiktoken


# Store the input in a tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])


# Split the data into training and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(train_data[:1000])
print(val_data[:1000])


# The model is not typically trained on all the data at once since it is computationally expensive
# The model will look at a block_size number of characters and predict the next character
# This is a simple way to handle long sequences
block_size = 8 # block_size is the number of characters in the input sequence
print(train_data[:block_size+1])

# NOTE:
# When you sample a chunk of characters like above, it actually has multiple examples
# for the model to learn from. For example, if you have the sequence "hello", the model
# will see the following examples:
# "hell" -> "o"
# "ell" -> "o"
# "ll" -> "o"
# "l" -> "o"

# For illustration
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")


# batch_size is the number of examples in the batch
# This is done for efficiency because GPUs are great at parallel processing
batch_size = 4
# block_size is maximum context length for predictions 
block_size = 8

# Create a batch of data
def get_batch(split:str) -> 'tuple[torch.Tensor, torch.Tensor]':
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print(xb.shape, yb.shape)
print(xb)
print(yb)

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context} the target: {target}")


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
print(logits)
# loss should be better than -ln(1/(vocab_size)) # why?


# generate new tokens
idx = torch.zeros((1,1), dtype=torch.long) # (B, T) is (1, 1) here. 
generated_tokens = m.generate(idx, max_new_tokens=30) # torch.Size([1, 31]
print(generated_tokens, generated_tokens.shape)

decoded_tokens = decode(generated_tokens[0].tolist())
print(decoded_tokens)



# Training the model

# create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

step_size = 10000
batch_size = 32
for steps in range(step_size):

    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# generate new tokens again
idx = torch.zeros((1,1), dtype=torch.long) # (B, T) is (1, 1) here. 

def generate_and_decode(model, idx, num_tokens=30):
    idx = torch.zeros((1,1), dtype=torch.long)
    generated = model.generate(idx, max_new_tokens=num_tokens)[0]
    return decode(generated.tolist())

# Usage
print(generate_and_decode(m, idx))

