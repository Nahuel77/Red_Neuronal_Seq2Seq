<h1>Sequência a sequência (seq2seq: sequence to sequence)</h1>

https://youtu.be/iynoMdzFmpc?si=Nat_KXNrJErmxXkE

Forward separado por etapas (duas sequências) que são passadas novamente por outro forward.

Antes tenho que dizer que neste código se deixa muito para as bibliotecas. Algo que evitei nas redes anteriores, usando bibliotecas que não me dessem a estrutura relativa à rede.

Mas tendo aprendido MLP, CNN, RNN (LTSM incluído), posso me permitir usar Torch (uma biblioteca que automatiza vários processos e nos oferece estruturas incluídas).

O que pode jogar a favor ou contra, depende de como é o programador. Eu costumo olhar mais o código por mim mesmo e evitá-las. Pelo menos no que é aprender.

Por isso, aqui neste resumo, não só explicarei lógica e estrutura, mas também os pontos que importam no uso da biblioteca.
Também não deixarei no código a etapa de saídas. Para aprendê-las, basta ver os outputs da etapa de aprendizado.

Por minha parte, jamais tinha usado Torch. Me parece que é uma arma de dois gumes. Oculta código demais para o meu gosto. E não imagino o quão trabalhoso pode se tornar se chegar a nos dar algum problema.
Olhando um pouco seu código-fonte, para o meu gosto, não está bem documentado.
Mas isso não tira que, sem Torch, teríamos que escrever muito mais código, que em parte já aprendemos com RNN e LSTM.

De novo, como em RNN, é difícil desenvolver um projeto “toy” que envolva uma tecnologia pensada para processar quantidades imensas de dados e que reflita bem o poder do que tentamos aprender.
Não é o que faz, mas como faz.

Este Seq2Seq tem como propósito prever uma sequência de números aprendida previamente.
De modo que, se lhe dermos um número, aprenda qual sequência vem a seguir.

Imaginemos que nossa rede está treinada com tokens de dígitos numéricos do estilo

    [[7,2,8,4,3], [6,9,4,7,1], [2,5,1,8,5], ...]

Passa-se um primeiro dado à rede para que ela preveja a sequência correta.
A tal dado se denomina SOS (Start of Sequence).
Suponhamos que o SOS é 7. A rede inferirá, pelo que aprendeu, que vem um 2; e se veio um 7 e depois um 2, inferirá que vem um 8... e assim por diante.

E se no batch houverem dois ou mais tokens que começam com 7?

    [[7,2,8,4,3], [7,6,5,2,8], [7,9,1,2,4], ...]

Lembremos que, igual à RNN, esta é uma rede pensada para trabalhar com sequências. A saída será a que tiver mais peso, segundo seu aprendizado. Mas, em um uso real desse tipo de rede, podemos usar as saídas com mais pesos para dar opções e ter um preditor de texto como os que usamos nos celulares.

A estrutura de treinamento é composta de 3 classes e é instanciada da seguinte maneira:

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)

Encoder e Decoder são instanciados independentemente. Mas Seq2Seq recebe ambos como parâmetros construtores.

Vejamos antes a configuração inicial:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 10
    seq_length = 5
    embedding_dim = 16
    hidden_dim = 32
    num_epochs = 2000
    batch_size = 64

device funciona como um switch: se existir GPU disponível, usará; caso contrário, trabalhará com a CPU.
Nosso vocabulário numérico tem tamanho 10 (números de 0 a 9). E o resto se explica por si só.

A rede toma o SOS e o representa em um vetor de pesos aleatórios.

Vejamos a classe Encoder:

    class Encoder(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        def forward(self, x):
            emb = self.embedding(x)
            outputs, (h, c) = self.lstm(emb)
            return h, c

Vemos que, no construtor, se inicia uma matriz de valores aleatórios de tamanho 10x16:

    self.embedding = nn.Embedding(vocab_size, embedding_dim)

Não vemos o uso de métodos como random, rand ou similares. nn.Embedding é um método que já se encarrega disso, pois Torch é uma biblioteca preparada para esse tipo de trabalho.

Isso nos dá um array para cada token do vocabulário. Cada array tem 16 valores.

    [[ 0.03, -0.21,  0.11, ...,  0.05],  # embedding para o <SOS> 0
     [ 0.12,  0.01, -0.07, ...,  0.08],  # embedding para o <SOS> 1
     ...
     [ 0.09, -0.14,  0.02, ..., -0.03]]  # embedding para o <SOS> 9

Portanto, ao passar o SOS para o forward, ele seleciona o embedding do token x.

    emb = self.embedding(x)

Se nosso SOS fosse 1, estaria representado por self.embedding(1) com [0.12, 0.01, -0.07, ..., 0.08]

Depois passa esse embedding pela LSTM da Torch e retorna suas saídas “h” e “c” (ver o repositório RNN e LSTM linkado no final deste README).

    outputs, (h, c) = self.lstm(emb)
    return h, c

Ainda que aqui não possamos ver porque usamos uma biblioteca. Em LSTM:
h → novo estado oculto (saída ativada, usada para predição).  
c → o novo estado interno (memória).
Também é retornado outputs, mas ainda não precisamos de nenhuma inferência, portanto omitimos seu uso. Retornamos apenas h e c.

No Decoder temos processos similares, mas com algumas diferenças...

Adicionamos uma transformação linear Wx + b (ver MLP), que se inicia com valores aleatórios.

    self.fc = nn.Linear(hidden_dim, vocab_size)

Aqui tive uma enorme confusão (por isso não gosto tanto das bibliotecas, embora seja culpa minha por querer ir rápido). nn.Linear não é uma transformação linear vista como uma simples multiplicação de matrizes. É mais que isso. É uma instanciação de uma classe Linear e, ao escrever a seguinte linha:

    logits = self.fc(outputs)

Executamos outro forward, que é o verdadeiro x*W_t + b. Simplificando: multiplicamos outputs pela transformação linear self.fc. Seria bom parar para estudar Torch a fundo. Mas não estou aprendendo bibliotecas, e sim redes neurais. De momento fico olhando redes.

    logits = self.fc(outputs)
    é equivalente a:
    logits = torch.matmul(outputs, self.fc.weight.T) + self.fc.bias
    Ou seja, x * W + b

A transformada dos pesos weight.T e os bias são próprios da classe e não da instância. Eles se iniciam com valores que serão treinados.

Decoder finalmente retorna logits, h e c:

    return logits, h, c

Vai ficando claro por que se chama Sequência a Sequência. Encoder para Decoder.

Antes de seguir com a explicação, comento um dado interessante que vi em um vídeo. Em 2016 o Google implementou NMT (Neural Machine Translator) no seu tradutor. E foi então que o tradutor começou realmente a funcionar com a efetividade que conhecemos hoje. Antes disso, não era tão bom.
Originalmente pensei em fazer um tradutor como projeto. Mas assumi que o dataset e o treinamento poderiam ser excessivos para meus recursos de hardware (um notebook com GPU limitada de motherboard).

Porém pensemos no exemplo do tradutor para entender melhor como seq2seq trabalha.

Encoder recebe uma frase em espanhol. Por exemplo "Hola mundo". Gera as saídas próprias da LSTM h e c.
Decoder recebe essa informação e produz as saídas. A classe Seq2Seq é responsável por gerir essas passagens, entre outras funções como calcular a perda.

        Encoder LSTM    ----c--->       Decoder LSTM
    {[Hola] -> [Mundo]} ----h---> {[SOS]:Hello -> [Hello]:World} --h--c--globals-->

Como se observa na classe Seq2Seq, o construtor recebe tanto encoder quanto decoder e os instancia como self.

    self.encoder = encoder
    self.decoder = decoder

No nosso ciclo for de épocas:

    for epoch in range(num_epochs):
        X, Y = generate_batch(batch_size, seq_length, vocab_size)

O que faz generate_batch():

    def generate_batch(batch_size, seq_length, vocab_size):
        X = np.random.randint(1, vocab_size, (batch_size, seq_length))
        Y = X.copy()
        return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

Declara X como uma matriz de números aleatórios inteiros que vai de 1 a 9 (1 a vocab_size) e cujo tamanho é 64x5 (batch_size, seq_length). Copia X para Y e retorna ambos como tensores Torch.

Temos a declaração dos batches X e Y. São similares, e um corresponde à entrada, outro à saída esperada. Conceito já visto em outras redes.
Depois passamos os batches ao model, que é equivalente a passá-los ao forward.

    output = model(X, Y)
    é equivalente a:
    output = model.forward(X, Y)

Porque assim funciona Torch :/

Vendo a classe observamos que forward recebe tais parâmetros como src e trg:

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
            input = trg[:, 0].unsqueeze(1)

            for t in range(1, trg_len):
                output, h, c = self.decoder(input, h, c)
                outputs[:, t] = output.squeeze(1)
                top1 = output.argmax(2)
                input = trg[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else top1

            return outputs

Forward cria uma matriz de zeros de tamanho (batch_size, trg_len, vocab_size) chamada outputs.
Depois enviamos o batch ao forward do encoder e recebemos h e c.

    h, c = self.encoder(src)

trg (que é o batch gerado no loop de treinamento e de forma 64x5) é dividido em batch_size. Ou seja, em grupos de 64 tokens:

    [[3],[3],[9],[2],[4],[2],[9],[5]...[8],[1],[5],[1],[9],[9],[3],[1]]

Isso é enviado ao Decoder tomando o primeiro como SOS. No loop seguinte vemos que, junto com o batch, enviamos h (como o Encoder memorizou os dados) e c (o que aprendeu a esquecer, reter ou ignorar — ver LSTM compuerta output).

    for t in range(1, trg_len):
        output, h, c = self.decoder(input, h, c)
        outputs[:, t] = output.squeeze(1)

Vemos que se define output, h e c (para o escopo da classe Seq2Seq), que é o que o Decoder processou. E começamos a empacotar os outputs em grupos de 5. Cada um desses está agrupado em 10 valores reais (sobre os quais se faz a inferência). E por sua vez agrupados em 64 (batch_size).

    [[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
      [-0.7767,  0.0528,  0.3603,  ..., -0.0651,  0.1891,  0.0400],
      ... x64

Finalmente se calcula a perda e atualizamos os pesos com o backward automatizado da Torch:

    loss = criterion(output[:, 1:].reshape(-1, vocab_size), Y[:, 1:].reshape(-1))
    loss.backward()

Como a perda e o backward trabalham se definem antes do loop:

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

Acredito que não seja necessário explicar novamente neste README como funciona um Backward ou como se calcula Loss. Embora existam diferentes maneiras, entendendo conceitualmente o que fazem e como fazem, já podemos confiar essa parte à Torch. De qualquer forma podem ser vistos nos repositórios anteriores.

Finalmente ordenamos à Torch executar os passos predefinidos:

    optimizer.step()

E já estará aprendendo em cada epoch. Vemos que em cada epoch também limpa o optimizer:

    optimizer.zero_grad()

Realmente não creio que seja necessário seguir explicando o restante do código. São etapas como avaliação e, embora não esteja presente, poderiam vir outras como uso direto da rede treinada ou KPIs.

<h2>Attention</h2>

Honestamente não vou parar para explicar muito. Não acho necessário. Se você chegou até aqui, poderá seguir o fio do código. Eu não vou ensinar a analisar código alheio neste README. Cada um saberá ver por si.
Vejamos pelo início os passos de como eu olho. Porque não é nada de outro mundo. Uma classe a mais e uma passagem de dados. Mas antes quero destacar que Torch não tem nenhuma classe reservada para Attention. O que a torna, na minha opinião, mais fácil quando se trata de ler código.

    attn = Attention(hidden_dim)
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, attn)
    model = Seq2Seq(encoder, decoder).to(device)

Olhando o treinamento, se vê claramente que se inicia uma classe Attention, além do encoder, decoder e modelo. Um caminho é olhar para trás (o que faz Attention), mas tendo feito esse caminho antes, prefiro ir para frente.
A classe instanciada attn é enviada ao decoder como parâmetro. E decoder, por sua vez, ao Seq2Seq.

Já sabemos como o modelo funciona em Seq2Seq, mas se olharmos como o modelo usa o decoder, veremos que o novo parâmetro é encoder_outputs.

    encoder_outputs, (h, c) = self.encoder(src)

Então vamos ver o encoder:

    return outputs, (h, c)

Vemos que desta vez o encoder efetivamente retorna e usa outputs. Coisa que no Seq2Seq plano não fazia, pois outputs ficava apenas como variável inicializada e esquecida. encoder_outputs é isso.

Depois decoder avalia outputs com o self.attention recebido na classe:

    self.attention = attention

O decoder então executa Attention guardando o retornado em attn_weights:

    attn_weights = self.attention(hidden_last, encoder_outputs)

Antes de seguir com o forward no decoder, vejamos a classe Attention:

    class Attention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, hidden, encoder_outputs):
            src_len = encoder_outputs.size(1)

            hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

            energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
            attention = self.v(energy).squeeze(2)

            return F.softmax(attention, dim=1)

Inicializa com um parâmetro hidden_dim, que é apenas uma medida arquitetônica. Toma essa medida e instancia duas nn.Linear. E se olharmos o forward, recebe 2 parâmetros (self não conta), de onde vínhamos.

Então, no decoder, forward calculava os pesos em attn_weights e lembramos que tínhamos outputs recebidos do encoder (encoder_outputs) e hidden_last (h[-1]). Com isso executamos o forward de Attention.

Attention retorna F.softmax(attention, dim=1), que já vimos em outras redes, são valores para ajustes do backward.

Attention é outra camada, mas implementada como classe; aplicando tanh a h_t com src_len, que é apenas o tamanho dos outputs fatiados (5) definidos na medida arquitetônica seq_length. Consultei o ChatGPT porque não tomava diretamente de lá; supõe-se ser boa prática pegar a medida do output recebido (faz sentido em código desacoplado).

Enfim. Concatenam-se hidden e encoder_outputs com torch.cat() e envia-se a self.attn(), armazenando seu tanh em energy:

    energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

Isso, por sua vez, vai a self.v (uma nn.Linear sem bias). E reestrutura-se as dimensões com squeeze():

    attention = self.v(energy).squeeze(2)

Por fim retorna:

    return F.softmax(attention, dim=1)

Se analisarmos com calma, vemos que é simplesmente outra camada filtrando valores por segmentos de outputs, retornando-os com softmax para que sejam interpretados como ajustes de peso.

Vejamos .squeeze e .unsqueeze com um código simples:

    cad = torch.tensor([[1,2,3],[4,5,9],[6,7,8]])
    print(cad)
    print(cad.unsqueeze(1))
    print(cad.squeeze(1))

Saída:

    tensor([[1, 2, 3],
            [4, 5, 9],
            [6, 7, 8]])

    tensor([[[1, 2, 3]],
            [[4, 5, 9]],
            [[6, 7, 8]]])

    tensor([[1, 2, 3],
            [4, 5, 9],
            [6, 7, 8]])

unsqueeze agrupa tudo em uma nova dimensão []; squeeze desfaz essa agregação. Basicamente agrupamos para obter promedios aplicados em Attention. Explicado grosseiramente, mas suficiente para entender.

O decoder recebe attn_weights e sucede:

    context = torch.bmm(attn_weights, encoder_outputs)

torch.bmm significa Batch Matrix Multiplication — calcula o contexto, que é outra coleção de valores que diz ao modelo a que deve prestar mais atenção. Como? concatena context com embedding:

    rnn_input = torch.cat((emb, context), dim=2)

As saídas junto com h e c retornam com nn.LSTM a partir de rnn_input:

    outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

Calculam-se as predições, agrupadas com unsqueeze:

    prediction = self.fc(torch.cat((outputs.squeeze(1), context.squeeze(1)), dim=1))

Decoder finalmente retorna:

    return prediction.unsqueeze(1), hidden, cell, attn_weights

O resto já foi visto.

<h3>Final.</h3>

Acho que, se temos que observar o salto que esta rede deu comparada com RNN, fica claro que demonstra a importância do Forward... É o coração de uma rede.
Aqui temos 3 forwards LSTM diretos (no encoder, no decoder e no modelo). E 1 indireto, se considerarmos a transformação linear do decoder. Mas apenas 1 backward.
Isso deixa claro que os cálculos de aprendizado ocorrem no Forward. E no Backward, os ajustes desses aprendizados.

Também é chamativo que, embora o poder desta rede seja maior que o das anteriores, essencialmente ocorrem os mesmos cálculos W*x + b. O que muda são as estruturas que fazem essas transformações acontecerem.

Imagino, neste ponto da minha jornada aprendendo redes neurais, que se voltássemos a obter uma matriz cujos valores fossem calculados novamente com outro processo LSTM — ou seja, outra camada de forwards — que novo filtro poderíamos aplicar? Provavelmente já se aplica.

Matrizes... Suponho que vê-las geometricamente às vezes pode ajudar a acostumar a mente ao que está acontecendo por trás do código. Mas isso só depois que os conceitos já estão claros. Seja o que for, o dado que as redes representam são, no fim, matrizes de valores numéricos trocando dados com operações matemáticas.
Mas também é bom ver como o pensamento pode ser abstraído em representações numéricas. Acho que é chave separar uma coisa da outra e pensar assim o que está sendo programado aqui.

</br>
MLP: https://github.com/Nahuel77/Red_Neuronal_MLP</br>
RNN e LSTM: https://github.com/Nahuel77/Red_Neuronal_RNN_LSTM_incluido</br>
