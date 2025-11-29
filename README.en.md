<h1>Sequence to Sequence (seq2seq: sequence to sequence)</h1>

https://youtu.be/iynoMdzFmpc?si=Nat_KXNrJErmxXkE

Forward separated by steps. (two sequences) that are passed again through another forward.

First, I have to say that in this code a lot is left to the libraries. Something I avoided in the previous networks, using libraries that did not give me the relative structure of the network.

But having learned MLP, CNN, RNN (LSTM included) I can allow myself to use Torch, (a library that automates several processes and provides us with built-in structures).

This can be an advantage or disadvantage, depending on the programmer. I usually look more at the code by myself and avoid them. At least when it comes to learning.

So, here in this summary, I will not only explain logic and structure, but also the points that matter regarding library usage.  
I will also not leave the output stage in the code. To learn it, just seeing the outputs of the learning stage is enough.

For my part, I had never used Torch. I think it's a double-edged sword. It hides too much code for my taste. And I can’t imagine how laborious it could get if it ever causes a problem.  
Looking a bit at its source code, in my opinion, it is not well documented.  
But that does not change the fact that without Torch, we would have to write much more code, which in part we already learned with RNN and LSTM.

Again, as in RNN, it is difficult to develop a toy project that involves a technology designed to process huge amounts of data and reflect well the power of what we are trying to learn.  
It is not what it does, but how it does it.

This Seq2Seq aims to predict a sequence of numbers learned previously.  
So that if we give it a number, it learns which sequence follows.

Imagine our network is trained with numeric tokens like:

    [[7,2,8,4,3], [6,9,4,7,1], [2,5,1,8,5], ...]

We pass the first data to the network to predict the correct sequence.  
Such data is called SOS (Start of Sequence).  
Suppose SOS is 7. The network will infer from what it learned that 2 follows; and if there was a 7 and then a 2, it will infer that 8 follows… and so on.

And what if in the batch there were two or more tokens starting with 7?

    [[7,2,8,4,3], [7,6,5,2,8], [7,9,1,2,4], ...]

Remember that, just like in RNN, this is a network designed to work with sequences. The output will be the one with the most weight, according to its learning. But in a real practical use of this type of network, we can use the outputs with higher weights to give options and have a text predictor like the ones we use on phones.

The training structure consists of 3 classes and they are instantiated as follows:

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)

Encoder and Decoder are instantiated independently. But Seq2Seq receives both as constructor parameters.

Let's look at the initial setup:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 10
    seq_length = 5
    embedding_dim = 16
    hidden_dim = 32
    num_epochs = 2000
    batch_size = 64

`device` works as a switch: if a GPU is available, it will use it; otherwise, it operates with the CPU.  
Our vocabulary, numeric, has a size of 10 (numbers from 0 to 9). The rest are self-explanatory.

The network takes the SOS and represents it in a vector of random weights.

Let's look at the Encoder class:

    class Encoder(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        def forward(self, x):
            emb = self.embedding(x)
            outputs, (h, c) = self.lstm(emb)
            return h, c

We see in the constructor that a matrix of random values of size 10x16 is initialized:

    self.embedding = nn.Embedding(vocab_size, embedding_dim)

We don’t see the use of methods like `random`, `rand`, or similar. `nn.Embedding` is a method that already takes care of this, as Torch is a library prepared for this type of work.

This gives us an array for each vocabulary token in our network. Each array has 16 values.

    [[ 0.03, -0.21, 0.11, ..., 0.05],  # embedding for <SOS> 0
     [ 0.12,  0.01, -0.07, ..., 0.08],  # embedding for <SOS> 1
     ...
     [ 0.09, -0.14, 0.02, ..., -0.03]]  # embedding for <SOS> 9

So when we pass SOS (x) to forward, it selects the embedding for token x.

    emb = self.embedding(x)

If our SOS was 1, it would be represented by `self.embedding(1)` with `[ 0.12,  0.01, -0.07, ..., 0.08]`.

Then this embedding passes through Torch's own LSTM network and returns its outputs `h` and `c` (see the RNN and LSTM repo linked at the end of this README).

    outputs, (h, c) = self.lstm(emb)
    return h, c

Although here we cannot see it because we use a library. In LSTM:  
`h` → new hidden state (activated output, used for prediction).  
`c` → new internal state (memory).  
`outputs` is also returned but we don’t need any inference yet, so its use is omitted. Only `h` and `c` are returned.

In the Decoder, we have similar processes but with some differences…

We add a linear transformation Wx + b (See MLP), initialized with random values.

    self.fc = nn.Linear(hidden_dim, vocab_size)

Here, I had a huge confusion (which is why I don’t like libraries much, although it’s my fault for wanting to go fast). `nn.Linear` is not just a linear transformation seen as a matrix multiplication. It is more than that. It is an instantiation of a Linear class, and when we later write:

    logits = self.fc(outputs)

We execute another forward which is the real x*W_t + b. Simplifying, we multiply the outputs by the linear transformation `self.fc`. Here it would be good to study Torch in depth. But I’m not learning libraries, I’m learning neural networks. For now, I focus on networks.

    logits = self.fc(outputs)
    is equivalent to:
    logits = torch.matmul(outputs, self.fc.weight.T) + self.fc.bias
    That is x*W + b

The weights `weight.T` and biases are part of the class and not the instance. They are initialized with values that will be trained.

The Decoder finally returns logits, h, and c:

    return logits, h, c

We can now see why it’s called Sequence to Sequence. Encoder to Decoder.

Before continuing, I want to comment an interesting fact I saw in a video. In 2016 Google implemented NMT (Neural Machine Translator) in its translator. That’s when Google Translate really started working with the effectiveness we know now. Before that, it was not a very good translator.  
Originally, I planned to make a translator as a project. But I assumed the dataset and training could be excessive for my hardware resources (a notebook with limited GPU).

However, let's think about the translator example to better understand how seq2seq works.

Encoder will receive a phrase in Spanish. For example "Hola mundo". It generates LSTM outputs h and c.  
Decoder receives this information and produces the outputs. The Seq2Seq class manages these changes among other functions like calculating loss.

        Encoder LSTM    ----c--->       Decoder LSTM
    {[Hola] -> [Mundo]} ----h---> {[SOS]:Hello -> [Hello]:World} --h--c--globals-->

As seen in the Seq2Seq class, the constructor receives both encoder and decoder and stores them as self.

    self.encoder = encoder
    self.decoder = decoder

In our epoch loop:

    for epoch in range(num_epochs):
        X, Y = generate_batch(batch_size, seq_length, vocab_size)

What does `generate_batch()` do?

    def generate_batch(batch_size, seq_length, vocab_size):
        X = np.random.randint(1, vocab_size, (batch_size, seq_length)) # 
        Y = X.copy()  # output same as input
        return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

It declares X as a matrix of random integers from 1 to 9 (1 to vocab_size) sized 64x5 (batch_size, seq_length). Copies X to Y and returns them as Torch tensors.

We have the batch declaration X and Y. They are similar: one is input, the other expected output for training. Concept already seen in other networks.  
Then we pass the batches to the model, which is equivalent to passing them to forward.

    output = model(X, Y)
    is equivalent to:
    output = model.forward(X, Y)
    Because that’s how Torch works :/

Looking at the class, we see that forward receives parameters like src and trg:

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, src, trg, teacher_forcing_ratio=0.5):
            batch_size, trg_len = trg.shape
            vocab_size = self.decoder.fc.out_features
            outputs = torch.zeros(batch_size, trg_len, vocab_size).to(device)

            h, c = self.encoder(src)
            input = trg[:, 0].unsqueeze(1)  # first token (could be <SOS>)

            for t in range(1, trg_len):
                output, h, c = self.decoder(input, h, c)
                outputs[:, t] = output.squeeze(1)
                top1 = output.argmax(2)
                input = trg[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else top1

            return outputs

Forward creates a zero matrix of size (batch_size, trg_len, vocab_size) called outputs.  
Then we send the batch generated in training to the encoder forward and receive h and c.

    h, c = self.encoder(src)

`trg` (the batch generated in the training loop, 64x5) is broken down by batch_size. That is, in groups of 64 tokens.

    [[3],[3],[9],[2],[4],[2],[9],[5]...[8],[1],[5],[1],[9],[9],[3],[1]]

The Decoder receives this, taking the first as SOS. In the next loop, together with batches, we send h (information about how the Encoder memorized the data) and c (what it learned to forget, retain, or ignore — see LSTM output gate).

    for t in range(1, trg_len):
        output, h, c = self.decoder(input, h, c)
        outputs[:, t] = output.squeeze(1)

We define output, h, and c (for the scope of Seq2Seq) which is what Decoder processed and start packing outputs in groups of 5. Each is grouped in 10 real values from which inference is done, and grouped in batches of 64.

    [[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
    [-0.7767,  0.0528,  0.3603,  ..., -0.0651,  0.1891,  0.0400],
    ...
    [-0.8650,  0.2022,  0.1934,  ..., 0.1269, 0.1168, 0.2491]],...]x64

Finally, loss is calculated and weights are updated with Torch's automated backward.

    loss = criterion(output[:, 1:].reshape(-1, vocab_size), Y[:, 1:].reshape(-1))
    loss.backward()

How loss is calculated and backward works is predefined before the for loop:

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

I don’t think it’s necessary to explain again in this README how Backward or Loss works. Conceptually understanding what they do (and how if possible) is enough; we can leave the rest to Torch. See previous repos for more.

Finally, we tell Torch to execute the preset steps:

    optimizer.step()

And learning happens in each epoch. In each epoch it also clears the optimizer:

    optimizer.zero_grad()

I really don’t think it’s necessary to explain the rest of the code. Stages like evaluation or using the trained network, or KPIs could follow.

<h2>Attention</h2>

Honestly, I won’t go deep into explaining it. If you’ve read this far, you can follow the code. I won’t teach how to analyze others’ code in this README. Everyone can see it as they wish.  
Let’s look at the steps from my perspective. It’s nothing extraordinary. One more class and a data pass. But first, Torch has no reserved Attention class. This makes it easier to read the code in my opinion.

    attn = Attention(hidden_dim)
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, attn)
    model = Seq2Seq(encoder, decoder).to(device)

Looking at training, we see the Attention class instantiated, along with encoder, decoder, and model. One approach is to look backward to see what Attention does, but having done that, it’s better to look forward.  
The instantiated class `attn` is passed as a parameter to the decoder instantiation, and decoder is passed to the Seq2Seq model.

We know how Seq2Seq works, but if we look at how the model uses the decoder, we see the new parameter `encoder_outputs`.

    encoder_outputs, (h, c) = self.encoder(src)

So let’s look at the encoder.

    return outputs, (h, c)

This time, the encoder actually returns and uses `outputs`. In plain Seq2Seq it did not, `outputs` was just a variable initialized and forgotten. `encoder_outputs` is that.  
Then decoder evaluates outputs using `self.attention` received as a class.

    self.attention = attention

Decoder executes Attention, storing returned values in `attn_weights`.

    attn_weights = self.attention(hidden_last, encoder_outputs)

Before continuing with decoder forward, let’s look at Attention class:

    class Attention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, hidden, encoder_outputs):
            src_len = encoder_outputs.size(1)

            # repeat hidden by src_len to concatenate
            hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

            # compute "energies"
            energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
            attention = self.v(energy).squeeze(2)

            return F.softmax(attention, dim=1)

It initializes with a parameter, `hidden_dim`, which is just an architectural measure. It uses this to instantiate two linear transformations (Linear, Torch class) as self.  
Its forward receives 2 parameters (self is not counted).  

In decoder forward, we calculate weights in `attn_weights` and we had `encoder_outputs` and `hidden_last` (states h[-1]). With this, we execute Attention forward.

Attention returns `F.softmax(attention, dim=1)` — values for backward adjustments.

Attention is another layer implemented as a class; tanh is applied to h_t with `src_len`, which is the output size split (5) already defined in seq_length. ChatGPT confirmed it’s better to take output size from received outputs (good practice for decoupled code).

We concatenate the two values with `torch.cat()` into a single vector. Send it to `self.attn()` with its dimension, storing tanh in `energy`.

    energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

This is sent to `self.v`, a nn.Linear without bias. Dimensions are reshaped with `.squeeze()`.

    attention = self.v(energy).squeeze(2)

Attention class returns softmaxed attention:

    return F.softmax(attention, dim=1)

In short, it’s another layer filtering values by output segments, returning them with softmax for weight interpretation.

Review `.squeeze()` and `.unsqueeze()` with simple code:

    cad = torch.tensor([[1,2,3],[4,5,9],[6,7,8]])
    print(cad)
    print(cad.unsqueeze(1))
    print(cad.squeeze(1))

Console output:

    tensor([[1, 2, 3],
            [4, 5, 9],
            [6, 7, 8]])

    tensor([[[1, 2, 3]],
            [[4, 5, 9]],
            [[6, 7, 8]]])

    tensor([[1, 2, 3],
            [4, 5, 9],
            [6, 7, 8]])

`unsqueeze` groups into a new `[]`, `squeeze` reverts. Basically, we group to compute Attention averages.

Decoder receives `attn_weights` and:

    context = torch.bmm(attn_weights, encoder_outputs)

`torch.bmm` (Batch Matrix Multiplication) computes context — another collection of values telling the model where to pay attention. It is concatenated with embedding:

    rnn_input = torch.cat((emb, context), dim=2)

Outputs and states h and c are passed through nn.LSTM with rnn_input:

    outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

Predictions are computed and grouped with `.unsqueeze()` in the return:

    prediction = self.fc(torch.cat((outputs.squeeze(1), context.squeeze(1)), dim=1))

Decoder finally returns:

    return prediction.unsqueeze(1), hidden, cell, attn_weights

The rest has been explained.

<h3>Final.</h3>

I think the leap this network makes compared to RNN is clear: it shows the importance of Forward… the heart of a network.  
Here we have 3 direct LSTM forwards (encoder, decoder, and model), and 1 more indirectly if we consider the linear transformation in decoder. But only 1 backward.  
This clearly shows that learning calculations occur in Forward; Backward adjusts the learned weights.  
Also notable: although this network is more powerful than its predecessors, essentially the same calculations W*x + b occur. The difference is how we structure these transformations.  
I imagine, at this point in learning neural networks, if we obtained a matrix whose values were recalculated by another LSTM process — another forward layer — what new filter could we apply? Probably already applied.

Matrices… Viewing it geometrically can help understand what happens behind the code. Once concepts are understood, the data networks work with are matrices exchanging numerical values through mathematical operations.  
It’s also good to see how thought can be abstracted into numerical representations. Separating one from the other is key in understanding the programming here.

</br>
MLP: https://github.com/Nahuel77/Red_Neuronal_MLP</br>
RNN and LSTM: https://github.com/Nahuel77/Red_Neuronal_RNN_LSTM_incluido</br>
