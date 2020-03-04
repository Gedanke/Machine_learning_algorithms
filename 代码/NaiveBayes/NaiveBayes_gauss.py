# -*- coding: utf-8 -*

import numpy
import pandas


class NaiveBayes(object):
    def __init__(self, path, rate):
        """
        :param path: csv文件路径(第一行不为列名)
        :param rate: 划分比例(以该比例划分数据集为训练集和测试集)
        """
        self.data = pandas.read_csv(path, header=None)
        self.rate = rate
        ''''''
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None
        self.init_data()
        self.num_samples = None
        self.num_class = None
        self.label_list = list()
        self.prior_prob = list()
        self.sample_mean = list()
        self.sample_var = list()

    def init_data(self):
        """
        :return:
        """
        '''去除含0的行'''
        lists = len(self.data.iloc[0])
        self.data.iloc[:, 0:lists - 1] = self.data.iloc[:, 0:lists - 1].applymap(
            lambda x: numpy.NAN if x == 0 else x)
        self.data = self.data.dropna(how="any")
        '''划分数据集'''
        train_data_ = self.data.sample(frac=self.rate, random_state=4, axis=0)
        test_index = [i for i in self.data.index.values if i not in train_data_.index.values]
        test_data_ = self.data.loc[test_index, :]
        '''得到训练数据及其标签'''
        self.train_data = train_data_.iloc[:, :-1]
        self.train_label = train_data_.iloc[:, -1]
        '''得到测试数据及其标签'''
        self.test_data = test_data_.iloc[:, :-1]
        self.test_label = test_data_.iloc[:, -1]

    def separate(self):
        """
        :return: data_class
        """
        self.num_samples = len(self.train_data)
        self.train_label = self.train_label.reshape(self.train_data.shape[0], 1)
        '''特征与标签合并'''
        data = numpy.hstack((self.train_data, self.train_label))
        data_class = dict()
        '''提取各类别数据,字典的键为类别名,值为对应的分类数据'''
        for index in range(len(data[:, -1])):
            if index in data[:, -1]:
                data_class[index] = data[data[:, -1] == index]
        self.train_label = numpy.asarray(self.train_label, numpy.float32)
        self.label_list = list(data_class.keys())
        self.num_class = len(data_class.keys())
        return data_class

    def gain_prior_prob(self, sample_label):
        """
        :param sample_label:
        :return:
        """
        return (len(sample_label) + 1) / (self.num_samples + self.num_class)

    def gain_sample_mean(self, sample):
        """
        :param sample:
        :return:
        """
        sample_mean = list()
        for index in range(sample.shape[1]):
            sample_mean.append(numpy.mean(sample[:, index]))
        return sample_mean

    def gain_sample_var(self, sample):
        """
        :param sample:
        :return:
        """
        sample_var = list()
        for index in range(sample.shape[1]):
            sample_var.append(numpy.var(sample[:, index]))
        return sample_var

    def gain_prob(self, sample, mean, var):
        """
        :param sample:
        :param mean:
        :param var:
        :return:
        """
        prob = list()
        for x, y, z in zip(sample, mean, var):
            prob.append((numpy.exp(-(x - y) ** 2 / (2 * z))) * (1 / numpy.sqrt(2 * numpy.pi * z)))
        return prob

    def train_model(self):
        """
        :return:
        """
        self.train_data = numpy.asarray(self.train_data, numpy.float32)
        self.train_label = numpy.asarray(self.train_label, numpy.float32)
        '''数据分类'''
        data_class = self.separate()
        '''计算各类别数据的目标先验概率,特征平均值和方差'''
        for data in data_class.values():
            sample = data[:, :-1]
            sample_label = data[:, -1]
            self.prior_prob.append(self.gain_prior_prob(sample_label))
            self.sample_mean.append(self.gain_sample_mean(sample))
            self.sample_var.append(self.gain_sample_var(sample))

    def predict(self, sample):
        """
        :return:
        """
        sample = numpy.asarray(sample, numpy.float32)
        poster_prob = list()
        idx = 0
        for x, y, z in zip(self.prior_prob, self.sample_mean, self.sample_var):
            gaussian = self.gain_prob(sample, y, z)
            poster_prob.append(numpy.log(x) + sum(numpy.log(gaussian)))
            idx = numpy.argmax(poster_prob)
        return self.label_list[idx]

    def get_result(self):
        """
        :return:
        """
        self.train_model()
        acc = 0
        tp = 0
        fp = 0
        fn = 0
        for index in range(len(self.test_data)):
            '''对self.test_data 进行预测'''
            predict = self.predict(self.test_data.iloc[index, :])
            target = numpy.array(self.test_label)[index]
            if predict == 1 and target == 1:
                tp += 1
            if predict == 0 and target == 1:
                fp += 1
            if predict == target:
                acc += 1
            if predict == 1 and target == 0:
                fn += 1
        return acc / len(self.test_data), tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)


if __name__ == "__main__":
    file_path = "pima-indians-diabetes.data.csv"
    rate_ = 0.8
    naiveBayes = NaiveBayes(file_path, rate_)
    acc_, tp_, fp_, fn_ = naiveBayes.get_result()
    print("准确率:", acc_)
    print("查准率:", tp_)
    print("查全率:", fp_)
    print("F1:", fn_)
