# библиотеки для работы с визуализацией
import seaborn as sns
import altair as alt
from matplotlib import pyplot as plt

#библиотеки для работы с системой и log, csv и time
import os
import logging
import  csv
import  time
from datetime import datetime

# для работы с массивами из С++
import numpy as np
# сигмоид expit()
import scipy.special

#базовые настройки log
logging.basicConfig(
    level= logging.DEBUG,
    filename= 'result_simple_nn.log')

logging.info('Start program simple_nn_2')

start_time = datetime.now() # получение время  начала выполненние программы без базовых настр.


'''Функции для проверки чтение, принимает два параметра
  Первый параметр file_to_read - имя файла csv которое нужно прочесит 
  Второй параметр var - вариант 1 - train.csv 2 - test.csv
  Возвращает построчно прочитонные csv 
  Все результаты записанные в log 
 '''
def csv_open(file_to_read, var = 1):
    file_to_read = csv.DictReader(file_to_read, delimiter = ',')
    if os.stat(file_to_read).st_size == 0:
        print('Csv file empth')
        logging.error('Файл csv пустой')
    else:
         if var == 1:
             result = open('mnist_train.csv', 'r')
             result_list = result.readlines()
             return  result_list
             result.close()
             logging.info('Функции открытие прочитала файл и вернула результат')
         elif var == 2:
             result = open('mnist_test.csv.csv', 'r')
             result_list = result.readlines()
             return  result_list
             result.close()
             logging.info('Функции открытие прочитала файл и вернула результат')



''' Функции генерации эпохи, принимает целое значение, которое является диапозоном от 0 до epochs 
    Возвращает так же целое число, по умолчанию стоить 1 
    Все результаты записываються в log
'''
def generate_random_epochs(epochs = 1 ):
    if isinstance(epochs,int) and epochs > 20:
        return  'Слишком большое значение \nБудет очень долго обрабатывать'
        logging.info('Передался слишком большой параметр epochs: \(epochs)')
    elif isinstance(epochs, int) == True and  epochs >= 0 and epochs <= 20:
        return  np.random.randint(0, epochs)
        logging.info('Функция генерации случайного epochs сгенирировал \(epochs)')
    else:
        return 'Введены не правильные значение'
        logging.error('Функции приняла не правильные значение и не отработал')



'''Функции генерации рандомного уровня обучение 
   Принимает один параметр learning_rate -  может быть int  или float, по умолчанию learning_rate принимает значение 0.1
   Возвращает в зависимости от введеного параметра либо int либо  float
   Все результаты записываються в log
'''
def generate_random_learning_rate(learning_rate = 0.1):
    if isinstance(learning_rate,int) or isinstance(learning_rate,float):
        logging.info('Функции генерации случайного уровня обучение выполнела работу')
        if isinstance(learning_rate, int):
            return np.random.randint(0, learning_rate+1)
        elif isinstance(learning_rate,float):
            return np.random.uniform(0.0, learning_rate)
    else:
        logging.error('Функции генерации рандомного уровня обучение получила не правильные значение')
        return 'Не правильное значение была передано'



# описание класса нейронной сети
'''Класс нейронной сети
  Основные методы: __init__,  train, query 
    __init_ -  инкапсулированный метод начальной инициализации 
  Принимает такие параметры: self, inputnodes, hiddennodes, outputnodes, learningrate
  inputnodes - целочисленное значение, входящих слоев
  hiddennodes - целочисленное значение, скрытые слоя 
  outputnodes - целочисленное значение, выходные слои 
  learningrate - может принимать как целочисленные так и float значение,  для уровня обучение 
'''
class neuralNetwork:

    # инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        logging.info('Создался объект класса neuralNetwork')
        # задание количества узлов входного, скрытого и выходного слоя
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # связь весовых матриц, wih и who
        # вес внутри массива w_i_j, где связь идет из узла i в узел j
        # следующего слоя
        # w11 w21
        # w12 w22 и т д7
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # уровень обучения
        self.lr = learningrate

        # функция активации - сигмоид
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

        # обучение нейронной сети

    def train(self, inputs_list, targets_list):
        logging.info('Запущенна функция тренировки нейронной сети')
        # преобразование входного списка 2d массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # вычисление сигналов на входе в скрытый слой
        hidden_inputs = np.dot(self.wih, inputs)
        # вычисление сигналов на выходе из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # вычисление сигналов на входе в выходной слой
        final_inputs = np.dot(self.who, hidden_outputs)
        # вычисление сигналов на выходе из выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка на выходе (целевое значение - рассчитанное)
        output_errors = targets - final_outputs
        # распространение ошибки по узлам скрытого слоя
        hidden_errors = np.dot(self.who.T, output_errors)

        # пересчет весов между скрытым и выходным слоем
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))

        # пересчет весов между входным и скрытым слоем
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))
        pass

    # запрос к нейронной сети
    def query(self, inputs_list):

        # преобразование входного списка 2d массив
        inputs = np.array(inputs_list, ndmin=2).T

        # вычисление сигналов на входе в скрытый слой
        hidden_inputs = np.dot(self.wih, inputs)
        # вычисление сигналов на выходе из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # вычисление сигналов на входе в выходной слой
        final_inputs = np.dot(self.who, hidden_outputs)
        # вычисление сигналов на выходе из выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# Задание архитектуры сети:
# количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# уровень обучения
learning_rate = 0.03

'''Функция визуализации  гиперпараметров 
   Первый параметр value_visual - принимает значение int, сам параметр влияет на какой библиотекой будет визуализируванне результаты, по умолчанию 1 
        1 - matplotlib, 2 - seaborn, 3 - altair 
   Второй параметр value_paramets -  принимает значение int, параметр влияет на генерацию измененний параметры learnint_rate, epochs 
        1 - меняем learning_rate но не меняем epochs, 2 - меняем epochs  но не минаем learning_rate, 3 - меняем и epochs и learning_rate 
   Третий параметр  count_generate -  количество генериремых обучений,  принимает int, по умолчанию стоит 100 
   Четвертный параметр max_rate - принимает float, влияет на максимльный уровень learning_rate, по умолчанию стоит 0.1
   П'ятый параметр max_epochs - принимает int, влияет на максимальное значение epochs,  по умолчанию стоит 20 
   Все промежуточные результаты записываються в log 
   Результатов функции является график 
'''
def visual_result_on_parameters(value_visual = 1 , value_parametrs = 1, count_generate = 5, max_rate = 0.1, max_epochs = 20):
    if value_visual == 1:
        if value_parametrs == 1:
            result = []
            iterator = np.random.uniform(0, max_rate, count_generate)
            for i in range(1, count_generate):
                learning_rate = iterator[i]
                n = neuralNetwork(input_nodes,
                                  hidden_nodes,
                                  output_nodes,
                                  learning_rate)
                training_data_file = open("mnist_train.csv", 'r')
                training_data_list = training_data_file.readlines()
                training_data_file.close()


                epochs = 1
                for e in range(epochs):
                        # итерирование по всем записям обучающего набора
                        for record in training_data_list:
                            # разделение записей по запятым ','
                            all_values = record.split(',')
                            # масштабирование и сдвиг исходных данных
                            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                            # создание целевых  выходов
                            targets = np.zeros(output_nodes) + 0.01
                            # элемент all_values[0] является целевым для этой записи
                            targets[int(all_values[0])] = 0.99
                            n.train(inputs, targets)
                            pass
                        pass

                test_data_file = open("mnist_test.csv", 'r')
                test_data_list = test_data_file.readlines()
                test_data_file.close()
                scorecard = []

                for record in test_data_list:
                    # разделение записей по запятым ','
                    all_values = record.split(',')
                    # правильный ответ - в первой ячейке
                    correct_label = int(all_values[0])
                    # масштабирование и сдвиг исходных данных
                    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                    # получение ответа от нейронной сети
                    outputs = n.query(inputs)
                    # получение выхода
                    label = np.argmax(outputs)
                    # добавление в список единицы, если ответ совпал с целевым значением
                    if (label == correct_label):
                        scorecard.append(1)
                    else:
                        scorecard.append(0)
                        pass

                    pass



                scorecard_array = np.asarray(scorecard)

                result.append(scorecard_array.sum() / scorecard_array.size)
            plt.plot(result, label = 'performance')
            plt.xlabel('learning_rate')
            plt.ylabel('performance')
            plt.title('График изменнения learning_rate')
            plt.show()
            logging.info('Функции визуализации, визулизировала график изменнение leraning_rate')
        elif value_parametrs == 2:
            result = []
            iterator = np.random.randint(0, max_epochs, count_generate)
            for i in range(1, count_generate):
                if value_parametrs == 2:
                    for i in range(1, count_generate):
                        learning_rate = max_rate
                        epochs = iterator[i]
                        n = neuralNetwork(input_nodes,
                                          hidden_nodes,
                                          output_nodes,
                                          learning_rate)
                        training_data_file = open("mnist_train.csv", 'r')
                        training_data_list = training_data_file.readlines()
                        training_data_file.close()

                        for e in range(epochs):
                            # итерирование по всем записям обучающего набора
                            for record in training_data_list:
                                # разделение записей по запятым ','
                                all_values = record.split(',')
                                # масштабирование и сдвиг исходных данных
                                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                                # создание целевых  выходов
                                targets = np.zeros(output_nodes) + 0.01
                                # элемент all_values[0] является целевым для этой записи
                                targets[int(all_values[0])] = 0.99
                                n.train(inputs, targets)
                                pass
                            pass

                        test_data_file = open("mnist_test.csv", 'r')
                        test_data_list = test_data_file.readlines()
                        test_data_file.close()
                        scorecard = []

                        for record in test_data_list:
                            # разделение записей по запятым ','
                            all_values = record.split(',')
                            # правильный ответ - в первой ячейке
                            correct_label = int(all_values[0])
                            # масштабирование и сдвиг исходных данных
                            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                            # получение ответа от нейронной сети
                            outputs = n.query(inputs)
                            # получение выхода
                            label = np.argmax(outputs)
                            # добавление в список единицы, если ответ совпал с целевым значением
                            if (label == correct_label):
                                scorecard.append(1)
                            else:
                                scorecard.append(0)
                                pass

                            pass

                        scorecard_array = np.asarray(scorecard)

                        result.append(scorecard_array.sum() / scorecard_array.size)
                    plt.plot(result, label='performance')
                    plt.xlabel('epochs')
                    plt.ylabel('performance')
                    plt.title('График изменнения epochs')
                    plt.show()
                    logging.info('Функции визуализации, визулизировала график изменнение epochs')
        elif value_parametrs == 3:

            result = []
            iterator_learning_rate = np.random.uniform(0, max_rate, count_generate)
            iterator_epochs = np.random.randint(0, max_epochs, count_generate)
            for i in range(1, count_generate):
                    learning_rate = iterator_learning_rate[i]
                    epochs = iterator_epochs[i]
                    n = neuralNetwork(input_nodes,
                                      hidden_nodes,
                                      output_nodes,
                                      learning_rate)
                    training_data_file = open("mnist_train.csv", 'r')
                    training_data_list = training_data_file.readlines()
                    training_data_file.close()


                    for e in range(epochs):
                        # итерирование по всем записям обучающего набора
                        for record in training_data_list:
                            # разделение записей по запятым ','
                            all_values = record.split(',')
                            # масштабирование и сдвиг исходных данных
                            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                            # создание целевых  выходов
                            targets = np.zeros(output_nodes) + 0.01
                            # элемент all_values[0] является целевым для этой записи
                            targets[int(all_values[0])] = 0.99
                            n.train(inputs, targets)
                            pass
                        pass

                    test_data_file = open("mnist_test.csv", 'r')
                    test_data_list = test_data_file.readlines()
                    test_data_file.close()
                    scorecard = []

                    for record in test_data_list:
                        # разделение записей по запятым ','
                        all_values = record.split(',')
                        # правильный ответ - в первой ячейке
                        correct_label = int(all_values[0])
                        # масштабирование и сдвиг исходных данных
                        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                        # получение ответа от нейронной сети
                        outputs = n.query(inputs)
                        # получение выхода
                        label = np.argmax(outputs)
                        # добавление в список единицы, если ответ совпал с целевым значением
                        if (label == correct_label):
                            scorecard.append(1)
                        else:
                            scorecard.append(0)
                            pass

                        pass

                    scorecard_array = np.asarray(scorecard)

                    result.append(scorecard_array.sum() / scorecard_array.size)
            plt.plot(result, label='performance')
            plt.xlabel('')
            plt.ylabel('performance')
            plt.title('График изменнения learning_rate и  epochs')
            plt.show()
            logging.info('Функции визуализации, визулизировала график изменнение learning_rate, epochs')
    elif value_visual == 2:
        if value_parametrs == 1:
            result = []
            iterator = np.random.uniform(0, max_rate, count_generate)
            for i in range(1, count_generate):
                learning_rate = iterator[i]
                n = neuralNetwork(input_nodes,
                                  hidden_nodes,
                                  output_nodes,
                                  learning_rate)
                training_data_file = open("mnist_train.csv", 'r')
                training_data_list = training_data_file.readlines()
                training_data_file.close()


                epochs = 1
                for e in range(epochs):
                        # итерирование по всем записям обучающего набора
                        for record in training_data_list:
                            # разделение записей по запятым ','
                            all_values = record.split(',')
                            # масштабирование и сдвиг исходных данных
                            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                            # создание целевых  выходов
                            targets = np.zeros(output_nodes) + 0.01
                            # элемент all_values[0] является целевым для этой записи
                            targets[int(all_values[0])] = 0.99
                            n.train(inputs, targets)
                            pass
                        pass

                test_data_file = open("mnist_test.csv", 'r')
                test_data_list = test_data_file.readlines()
                test_data_file.close()
                scorecard = []

                for record in test_data_list:
                    # разделение записей по запятым ','
                    all_values = record.split(',')
                    # правильный ответ - в первой ячейке
                    correct_label = int(all_values[0])
                    # масштабирование и сдвиг исходных данных
                    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                    # получение ответа от нейронной сети
                    outputs = n.query(inputs)
                    # получение выхода
                    label = np.argmax(outputs)
                    # добавление в список единицы, если ответ совпал с целевым значением
                    if (label == correct_label):
                        scorecard.append(1)
                    else:
                        scorecard.append(0)
                        pass

                    pass



                scorecard_array = np.asarray(scorecard)

                result.append(scorecard_array.sum() / scorecard_array.size)
            sns.distplot()
           #plt.plot(result, label = 'performance')
            plt.xlabel('learning_rate')
            plt.ylabel('performance')
            plt.title('График изменнения learning_rate')
            plt.show()
            logging.info('Функции визуализации, визулизировала график изменнение leraning_rate')
    elif value_visual == 3:
        if value_parametrs == 1:
            result = []
            iterator = np.random.uniform(0, max_rate, count_generate)
            for i in range(1, count_generate):
                learning_rate = iterator[i]
                n = neuralNetwork(input_nodes,
                                  hidden_nodes,
                                  output_nodes,
                                  learning_rate)
                training_data_file = open("mnist_train.csv", 'r')
                training_data_list = training_data_file.readlines()
                training_data_file.close()

                epochs = 1
                for e in range(epochs):
                    # итерирование по всем записям обучающего набора
                    for record in training_data_list:
                        # разделение записей по запятым ','
                        all_values = record.split(',')
                        # масштабирование и сдвиг исходных данных
                        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                        # создание целевых  выходов
                        targets = np.zeros(output_nodes) + 0.01
                        # элемент all_values[0] является целевым для этой записи
                        targets[int(all_values[0])] = 0.99
                        n.train(inputs, targets)
                        pass
                    pass

                test_data_file = open("mnist_test.csv", 'r')
                test_data_list = test_data_file.readlines()
                test_data_file.close()
                scorecard = []

                for record in test_data_list:
                    # разделение записей по запятым ','
                    all_values = record.split(',')
                    # правильный ответ - в первой ячейке
                    correct_label = int(all_values[0])
                    # масштабирование и сдвиг исходных данных
                    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                    # получение ответа от нейронной сети
                    outputs = n.query(inputs)
                    # получение выхода
                    label = np.argmax(outputs)
                    # добавление в список единицы, если ответ совпал с целевым значением
                    if (label == correct_label):
                        scorecard.append(1)
                    else:
                        scorecard.append(0)
                        pass

                    pass

                scorecard_array = np.asarray(scorecard)

                result.append(scorecard_array.sum() / scorecard_array.size)
            alt.Chart(result).mark_line()
            logging.info('Функции визуализации, визулизировала график изменнение leraning_rate')




#visual_result_on_parameters(max_rate= 0.3, value_visual= 1, max_epochs = 20, value_parametrs = 2 )








# создание экземпляра класса нейронной сети
n = neuralNetwork(input_nodes,
                  hidden_nodes,
                  output_nodes,
                  learning_rate)


# Загрузка тренировочного набора данных
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Обучение нейронной сети
# количество эпох
epochs = 20




for e in range(epochs):

    #итерирование по всем записям обучающего набора
    for record in training_data_list:
        # разделение записей по запятым ','
        all_values = record.split(',')
        # масштабирование и сдвиг исходных данных
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # создание целевых  выходов
        targets = np.zeros(output_nodes) + 0.01
        # элемент all_values[0] является целевым для этой записи
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# Загрузка тестового набора данных
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Тестирование нейронной сети

# Создание пустого накопителя для оценки качества
scorecard = []

# итерирование по тестовому набору данных
for record in test_data_list:
    # разделение записей по запятым ','
    all_values = record.split(',')
    # правильный ответ - в первой ячейке
    correct_label = int(all_values[0])
    # масштабирование и сдвиг исходных данных
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # получение ответа от нейронной сети
    outputs = n.query(inputs)
    # получение выхода
    label = np.argmax(outputs)
    # добавление в список единицы, если ответ совпал с целевым значением
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass

    pass

# расчет точности классификатора
scorecard_array = np.asarray(scorecard)
print('Получаем результат при learning_rate: {0}, epochs: {1} '.format(learning_rate,epochs))
print("performance = ", scorecard_array.sum() / scorecard_array.size)


end_time = datetime.now() #получение время окончание выполненние программы

result_time = end_time - start_time
print('Программа выполннила свою работу: ', result_time)
logging.info('Программа закончила свою работу за {0}'.format(str(result_time)))
