# -*- coding: utf-8 -*

import csv
import math
import random


def mean(feature):
    """
    :return: 特征均值
    """
    return sum(feature) / float(len(feature))


def sta_dev(feature):
    """
    :return: 特征标准差
    """
    mean_ = mean(feature)
    return math.sqrt(sum([pow(f - mean_, 2) for f in feature]) /
                     float(len(feature) - 1))


def summary(instances):
    """
    :return: summary_
    """
    summary_ = [(mean(feature), sta_dev(feature)) for feature in zip(*instances)]
    del summary_[-1]
    return summary_


def calculate_possibility(x, mean_, sta_dev_):
    """
    :param x:
    :param mean_:
    :param sta_dev_:
    :return:
    """
    exponent = math.exp(-(math.pow(x - mean_, 2) / (
            2 * math.pow(sta_dev_, 2))))
    return (1 / (math.sqrt(2 * math.pi) * sta_dev_)) * exponent


class NaiveBayes(object):
    def __init__(self, path, rate):
        """
        :param path: csv文件路径(第一行不为列名)
        :param rate: 划分比例(以该比例划分数据集为训练集和测试集)
        """
        self.path = path
        self.rate = rate
        self.data = list()
        self.train_data = list()
        self.test_data = list()
        self.separated_data = dict()
        self.summary_data = dict()
        self.probability = dict()
        '''函数'''
        self.init_data()
        self.separate_summary()

    def init_data(self):
        """
        读取并划分数据集
        :return:
        """
        '''读取'''
        all_lines = csv.reader(open(self.path, "r"))
        self.data = list(all_lines)
        size = len(self.data)
        for i in range(size):
            '''浮点化数据'''
            self.data[i] = [float(x) for x in self.data[i]]
        '''划分'''
        sample_num = int(size * self.rate)
        sample_index = random.sample(list(range(size)), sample_num)
        sorted(sample_index)
        self.test_data = list(self.data)
        for index in sample_index:
            self.train_data.append(self.test_data.remove(self.data[index]))

    def separate_summary(self):
        """
        按标签划分数据集
        :return:
        """
        for index in range(len(self.test_data)):
            sample = self.test_data[index]
            if sample[-1] not in self.separated_data:
                self.separated_data[sample[-1]] = []
            self.separated_data[sample[-1]].append(sample)
        for class_value, instances in self.separated_data.items():
            self.summary_data[class_value] = summary(instances)

    def calculate_possibility(self, sample):
        """
        :param sample:
        :return:
        """
        for classValue, classSummary in self.summary_data.items():
            self.probability[classValue] = 1
            for index in range(len(classSummary)):
                mean_, sta_dev_ = classSummary[index]
                x = sample[index]
                self.probability[classValue] *= calculate_possibility(x, mean_, sta_dev_)

    def predict(self, sample):
        """
        :return: best_label
        """
        self.calculate_possibility(sample)
        best_label = None
        best_pro = -1
        for classValue, probability in self.probability.items():
            if best_label is None or probability > best_pro:
                best_pro = probability
                best_label = classValue
        return best_label

    def get_predictions(self):
        """
        :return: predictions
        """
        predictions = []
        for index in range(len(self.test_data)):
            predictions.append(self.predict(self.test_data[index]))
        return predictions

    def get_result(self):
        """
        :return: 正确率
        """
        correct = 0
        predictions = self.get_predictions()
        for index in range(len(self.test_data)):
            if self.test_data[index][-1] == predictions[index]:
                correct += 1
        return correct / float(len(self.test_data))


if __name__ == '__main__':
    file_path = 'pima-indians-diabetes.data.csv'
    rate_ = 0.67
    naiveBayes = NaiveBayes(file_path, rate_)
    print("Result: {0}%".format(naiveBayes.get_result() * 100))
