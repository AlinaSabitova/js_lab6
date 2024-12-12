const csvUrl = 'https://gist.githubusercontent.com/juandes/2f1ffa32dd4e58f9f5825eca1806244b/raw/c5b387382b162418f051fd83d89fddb4067b91e1/steps_distance_df.csv';
// Local version of the dataset
// const csvUrl = 'steps_distance_df.csv'
const dataSurface = { name: 'Steps and Distance Scatterplot', tab: 'Data' };
const fittedSurface = { name: 'Fitted Dataset', tab: 'Data' };
const dataToVisualize = [];
const predictionsToVisualize = [];
let lastThreeLosses = []; // Хранилище для значений loss за последние 3 эпохи

let csvDataset;
let model;

async function defineAndTrainModel(numberEpochs) {
  // Make sure the tfjs-vis visor is open.
  tfvis.visor().open();

  const numOfFeatures = (await csvDataset.columnNames()).length - 1;

  const flattenedDataset = csvDataset
    .map(({ xs, ys }) => ({ xs: Object.values(xs), ys: Object.values(ys) }))
    .batch(32);

  model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    units: 1,
  }));

  model.compile({
    optimizer: tf.train.adam(0.1),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'], // Also mean squared error
  });

  const history = await model.fitDataset(flattenedDataset, {
    epochs: numberEpochs,
    callbacks: [
      tfvis.show.fitCallbacks(
        { name: 'Loss и MSE', tab: 'Training' },
        ['loss', 'mse'],
        { callbacks: ['onEpochEnd'] },
      ),
      {
        onEpochEnd: async (epoch, logs) => {
          console.log(`${epoch}: ${logs.loss}`);
          lastThreeLosses.push(logs.loss); // Сохраняем loss текущей эпохи
          if (lastThreeLosses.length > 3) {
            lastThreeLosses.shift(); // Удаляем самые старые значения
          }
        },
      }],
  });

  // Вычисление среднего значения loss за последние 3 эпохи
  if (lastThreeLosses.length > 0) {
    const averageLoss = (lastThreeLosses.reduce((sum, loss) => sum + loss, 0) / lastThreeLosses.length).toFixed(4);
    displayAverageLoss(averageLoss); // Вывод среднего значения
  }

  drawFittedLine(0, 30000, 500);
  model.summary();
  console.log(`Model weights:\n ${model.getWeights()}`);

  document.getElementById('predict-btn').disabled = false;
}

// Функция для отображения среднего значения loss
function displayAverageLoss(averageLoss) {
  let avgLossElem = document.getElementById('average-loss');
  if (!avgLossElem) {
    avgLossElem = document.createElement('p');
    avgLossElem.id = 'average-loss';
    document.querySelector('#train-div').appendChild(avgLossElem);
  }
  avgLossElem.innerText = `Среднее значение loss за последние 3 эпохи: ${averageLoss}`;
}

async function loadData() {
  // Our target variable (what we want to predict) is the the column 'distance'
  // so we add it to the configuration as the label
  csvDataset = tf.data.csv(
    csvUrl, {
      columnConfigs: {
        distance: {
          isLabel: true,
        },
      },
    },
  );


  await csvDataset.forEachAsync((e) => {
    dataToVisualize.push({ x: e.xs.steps, y: e.ys.distance });
  });

  tfvis.render.scatterplot(dataSurface, { values: [dataToVisualize], series: ['Dataset'] });
}

function createLoadPlotButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Загрузить и визуализировать данные';
  btn.id = 'load-plot-btn';

  // Listener that waits for clicks.
  // Once a click is done, it will execute the function
  btn.addEventListener('click', () => {
    loadData();
    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = false;
  });


  document.querySelector('#load-plot').appendChild(btn);
}

function createTrainButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Обучить!';
  btn.disabled = true;
  btn.id = 'train-btn';

  // Listener that waits for clicks.
  // Once a click is done, it will execute the function
  btn.addEventListener('click', () => {
    const numberEpochs = document.getElementById('number-epochs').value;
    defineAndTrainModel(parseInt(numberEpochs, 10));
  });

  document.querySelector('#train-div').appendChild(btn);
}

function drawFittedLine(min, max, steps) {
  // Empty the array in case the user trains more than once.
  const fittedLinePoints = [];
  const predictors = Array.from(
    { length: (max - min) / steps + 1 },
    (_, i) => min + (i * steps),
  );

  const predictions = model.predict(tf.tensor1d(predictors)).dataSync();

  predictors.forEach((value, i) => {
    fittedLinePoints.push({ x: value, y: predictions[i] });
  });

  const structureToVisualize = {
    values: [dataToVisualize, fittedLinePoints],
    series: ['1. Данные обучения', '2. Полученная зависимость'],
  };

  tfvis.render.scatterplot(fittedSurface, structureToVisualize);
}

function createPredictionInput() {
  const input = document.createElement('input');
  input.type = 'number';
  input.id = 'predict-input';

  document.querySelector('#predict').appendChild(input);
}

function createPredictionOutputParagraph() {
  const p = document.createElement('p');
  p.id = 'predict-output-p';

  document.querySelector('#predict').appendChild(p);
}

function createPredictButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Спрогнозировать!';
  btn.disabled = true;
  btn.id = 'predict-btn';

  // Listener that waits for clicks.
  // Once a click is done, it will execute the function
  btn.addEventListener('click', () => {
    // Получаем значение из ввода
    const valueToPredict = document.getElementById('predict-input').value;
    const parsedValue = parseInt(valueToPredict, 10);
    if (parsedValue < 0) {
      alert("Пожалуйста, введите неотрицательное значение.");
      return; // Выход из функции, если значение отрицательное
    }
  
    // Получаем результат предсказания и преобразуем его в число
    const prediction = model.predict(tf.tensor1d([parsedValue])).dataSync()[0]; // Берём только первое значение
  
    // Отображаем результат на экране
    const p = document.getElementById('predict-output-p');
    p.innerHTML = `Predicted value is: ${prediction}`;
  
    // Сохраняем значение шага и предсказания в массив
    predictionsToVisualize.push({ x: parsedValue, y: prediction });
  
    // Рисуем обновлённый график
    const structureToVisualize = {
      values: [dataToVisualize, predictionsToVisualize],
      series: ['1. Данные обучения', '2. Predictions'],
    };
    tfvis.render.scatterplot(dataSurface, structureToVisualize);
  
    // Автоматически переключаемся на вкладку "Data"
    tfvis.visor().setActiveTab('Data');
  });

  document.querySelector('#predict').appendChild(btn);
}

// Функция сохранения в localStorage
function createSaveResultsButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Сохранить результаты';
  btn.id = 'save-results-btn';

  btn.addEventListener('click', () => {
    // Сохраняем результаты в localStorage
    localStorage.setItem('predictions', JSON.stringify(predictionsToVisualize));
    alert('Результаты успешно сохранены в localStorage!');
  });

  document.querySelector('#save-results-div').appendChild(btn);
}

// Создание кнопки для отображения сохраненных результатов
function createShowSavedResultsButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Показать сохраненные результаты';
  btn.id = 'show-saved-results-btn';

  btn.addEventListener('click', () => {
    // Получаем данные из localStorage
    const savedPredictions = JSON.parse(localStorage.getItem('predictions'));

    // Проверяем, есть ли сохраненные данные
    if (!savedPredictions || savedPredictions.length === 0) {
      alert('Нет сохраненных результатов.');
      return;
    }

    // Получаем контейнер для отображения результатов
    const resultsContainer = document.querySelector('#saved-results');
    resultsContainer.innerHTML = ''; // Очищаем контейнер

    // Создаем заголовок
    const header = document.createElement('h3');
    header.innerText = 'Сохраненные результаты:';
    resultsContainer.appendChild(header);

    // Добавляем строки с результатами
    savedPredictions.forEach(({ x, y }, index) => {
      const resultParagraph = document.createElement('p');
      resultParagraph.style.margin = '5px 0';
      resultParagraph.innerText = `${index + 1}: Шаги = ${x}, Прогнозируемое расстояние = ${y.toFixed(2)}`;
      resultsContainer.appendChild(resultParagraph);
    });
  });

  document.querySelector('#save-results-div').appendChild(btn);
}

// Инициализация приложения
function init() {
  createTrainButton();
  createPredictionInput();
  createPredictButton();
  createPredictionOutputParagraph();
  createLoadPlotButton();
  createSaveResultsButton();
  createShowSavedResultsButton();
}

init();