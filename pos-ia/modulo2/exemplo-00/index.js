const tf = require('@tensorflow/tfjs');

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Primeira camada  da rede:
    // entrada de 7 posiçoes (idade normalizada + 3 cores + 3 localizações)
    // 80 neurônios na primeira camada -> tem pouca base de treino e por isso 
    // coloquei mais neuronios. Quando mais neuronios, mais complexa a rede e 
    // mais dados de treino são necessários e consequentemente mais processamento ela usa.
    // A ReLUU age como um filtro: é como se ela deixasse somente os dados interessantes 
    // seguirem viagem na rede. Se a info chegou nesse neuronio é positivo, passa para frente
    // Se for 0 ou negativo, descarta.
    model.add(tf.layers.dense({
        inputShape: [7],
        units: 80,
        activation: 'relu'
    }));

    // Saida: 3 neuronios por causa do one-hot encoding (premium, medium, basic)
    // A softmax é uma função de ativação que transforma os valores de saída em probabilidades
    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
    }));

    // Compilando o modelo  com o otimizador Adam que é um treinador pesssoal moderno para redesneurais 
    // Ele ajusta os pesos de forma eficiente e inteligente e aprende com o historico de erros e acertos
    // Loss: categoricalCrossentropy
    // Compara o que o modelo acha ( os scores de cada categoria) com o que é a verdade (o one-hot encoding) e calcula o erro.
    // Ex: a categoria premiom sempre [1,0,0] e ai ele vai treinando ate encontrar esse resultado
    // Quando  mais distante da precisao do modelo da resposta correta
    // maior sera o erro
    // Ex classico: classificacao de imagens, recomendacao e categorizacao
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Treinamento do modelo
    await model.fit(inputXs, outputYs, {
        verbose: 0, // Mostra o progresso do treino no console(desabilita o log interno e usa so o console log do callback)
        epochs: 100, // Quantas vezes o modelo vai ver os dados de treino
        shuffle: true, // Embaralha os dados a cada época para evitar overfitting
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
            }
        }
    });

    return model;
}

async function predict(model, pessoa) {
    // transformar o arrya para o tensor 2D
    const tfInput = tf.tensor2d(pessoa);

    // faz a predicao(output sera um vetor de 3 probabilidades)
    const prediction = model.predict(tfInput);
    const predictionArray = await prediction.array();
    //console.log(predictionArray)

    return predictionArray[0].map((probabilidade, index) => ({
        probabilidade,
        index
    }));
}


// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = 
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// Formula: idade_normalizada = (idade - idade_min) / (idade_max - idade_min)
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// inputXs.print();
// outputYs.print();

// quanto mais dado melhor o modelo, mais preciso ele fica. 
// Assim o algoritmo consegue aprender melhor a diferenciar as categorias.

const pessoaTeste = {
    nome: "Ze",
    idade: 28,
    cor: "verde",
    localizacao: "Curitiba"
}

// Normalizando a idade da nova pessoa usando o mesmo padrao do treino
//Ex: idade_min = 25, idade_max = 40 entaoo (28-25)/(40-25) = 0.2

const pessoaTensorNormalizada = [[
    0.2, // idade normalizada
    0,   // azul
    0,   // vermelho
    1,   // verde
    0,   // São Paulo
    1,   // Rio
    0    // Curitiba
]]

async function main() {
    const model = await trainModel(inputXs, outputYs);
    const predictions = await predict(model, pessoaTensorNormalizada);

    const result = predictions
        .sort((a, b) => b.probabilidade - a.probabilidade)
        .map(pred => `${labelsNomes[pred.index]} (${(pred.probabilidade * 100).toFixed(2)}%)`)
        .join('\n');

    console.log(`Predições para ${pessoaTeste.nome}:`, result);
}

main();
