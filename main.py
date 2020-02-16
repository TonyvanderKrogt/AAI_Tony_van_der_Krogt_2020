import numpy as np


class kNN:
    def __init__(self, k):
        self.k = k
        self.labels = []
        self.val_labels = []
        self.val_answers = []
        self.data = np.genfromtxt("C:/ti-software/AAAI/Opdrachten/3.1 kNN/dataset1.csv", delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
        self.dates = np.genfromtxt("C:/ti-software/AAAI/Opdrachten/3.1 kNN/dataset1.csv", delimiter=";", usecols=[0])
        self.validationdates = np.genfromtxt("C:/ti-software/AAAI/Opdrachten/3.1 kNN/validation1.csv", delimiter=";", usecols=[0])
        ##Label a season to a date through the datum of the date
        for label in self.dates:
            if label < 20000301:
                self.labels.append("winter")
            elif 20000301 <= label < 20000601:
                self.labels.append("lente")
            elif 20000601 <= label < 20000901:
                self.labels.append("zomer")
            elif 20000901 <= label < 20001201:
                self.labels.append("herfst")
            else:  # from 01-12 to end of year
                self.labels.append("winter")
        ##Label a season to a date through the datum of the date
        for label in self.validationdates:
            if label < 20010301:
                self.val_answers.append("winter")
            elif 20010301 <= label < 20010601:
                self.val_answers.append("lente")
            elif 20010601 <= label < 20010901:
                self.val_answers.append("zomer")
            elif 20010901 <= label < 20011201:
                self.val_answers.append("herfst")
            else:  # from 01-12 to end of year
                self.labels.append("winter")

    ##def validate_data
    ##path: File path of the dataset you want to test
    ## enters self.val_data with the day_values of your validation set. Then you will find for each point the closest neighbour and attach a season to it.
    def validate_data(self, path):
        self.val_data = np.genfromtxt(path, delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
        for day_val in self.val_data:
            neighbours = self.find_neighbours(day_val)
            seasons = [0,0,0,0]
            for i in neighbours:
                if self.labels[int(i)] == "winter":
                    seasons[0] += 1
                elif self.labels[int(i)] == "lente":
                    seasons[1] += 1
                elif self.labels[int(i)] == "zomer":
                    seasons[2] += 1
                else:
                    seasons[3] += 1
            season_choice = seasons.index(max(seasons))
            if season_choice == 0:
                self.val_labels.append("winter")
            elif season_choice == 1:
                self.val_labels.append("lente")
            elif season_choice == 2:
                self.val_labels.append("zomer")
            else:
                self.val_labels.append("herfst")
        self.create_mark()

    ##def create_mark
    ##return: Create a mark to check how many days are correctly filled
    def create_mark(self):
        good = 0
        for i in range(len(self.val_labels)):
            if(self.val_labels[i] == self.val_answers[i]):
                good += 1
        mark = good / len(self.val_labels) * 100
        print(str(self.k) + ": " + str(mark))

    ##def find_neighbours
    ##day_var: The centerpoint you want to know the distances of K neighbours
    ##return: Return K closest points of day_var
    def find_neighbours(self, day_var):
        neighbours = np.array([])
        differences = np.array([])
        for i in self.data:
            differences = np.append(differences, self.pythagoras(day_var, i))
        result = (np.argpartition(differences, self.k))
        for i in range(self.k):
            neighbours = np.append(neighbours, result[i])
        return neighbours
    ##def pythagoras
    ##start_point: Value of the starting point
    ##end_point: Value of the end point
    ##Return: Return distance between 2 points
    ## Get the distance between 2 points
    def pythagoras(self, start_point, end_point):
        diff = 0
        if len(end_point) == len(start_point):
            for i in range(len(start_point)):
                diff += (start_point[i] - end_point[i]) ** 2
        return diff


for i in range(200):
    start = kNN(i)
    start.validate_data("C:/ti-software/AAAI/Opdrachten/3.1 kNN/validation1.csv")
