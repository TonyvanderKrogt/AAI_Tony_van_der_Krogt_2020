import numpy as np
import random
import copy
import matplotlib.pyplot as plt

class kMeans:
    def __init__(self, k, path):
        self.k = k
        self.centroids = np.zeros(shape=(k, 7))
        self.labels = np.array([])
        self.all_K_used = np.array([])
        self.all_shortest_dis = np.array([])
        self.shortest_centroids = np.array([])

        self.data = np.genfromtxt(path, delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7],
                                        converters={5: lambda s: 0 if s == b"-1" else float(s),
                                                    7: lambda s: 0 if s == b"-1" else float(s)})
        self.dates = np.genfromtxt(path, delimiter=";", usecols=[0])
        self.dates = np.genfromtxt(path, delimiter=";", usecols=[0])
        self.min_max = np.full((2, len(self.data[0])), 0)
        self.centroid_choice = np.zeros(shape=(len(self.data)))
        self.cluster_season = [""] * self.k




        for label in self.dates:
            if label < 20000301:
                self.labels = np.append(self.labels, "winter")
            elif 20000301 <= label < 20000601:
                self.labels = np.append(self.labels, "lente")
            elif 20000601 <= label < 20000901:
                self.labels = np.append(self.labels, "zomer")
            elif 20000901 <= label < 20001201:
                self.labels = np.append(self.labels, "herfst")
            else:  # from 01-12 to end of year
                self.labels = np.append(self.labels, "winter")

    def empty_centroids(self):
        self.centroids = np.zeros(shape=(self.k, 7))

    ##create_centroids
    ##chooses K points from data set to start as centroids
    def create_centroids(self):
        data_copy = copy.deepcopy(self.data)
        np.random.shuffle(data_copy)
        for index_centroid in range(self.k):
            for index_day_val in range(len(self.data[0])):
                self.centroids[index_centroid][index_day_val] = data_copy[index_centroid][index_day_val]

    ##new_centroids
    ##day_val: The day you want to find out which centroid is the closest
    ##return: index of the closest centroid
    def closest_centroid(self, day_val):
        closest_centroid = self.pythagoras(self.centroids[0], day_val)
        closest_index = 0
        for centroid in range(len(self.centroids)):
            new_distance = self.pythagoras(self.centroids[centroid], day_val)
            if new_distance < closest_centroid:
                closest_centroid = new_distance
                closest_index = centroid
        return closest_index

    def print(self):
        print(self.cluster_season)

    def set_k(self, k):
        self.k = k

    def count_distance(self):
        sum_list = 0
        for day in range(len(self.data)):
            distance = self.pythagoras(self.centroids[int(self.centroid_choice[day])], self.data[day])
            sum_list += distance
        return sum_list


    def choose_new_centroid(self):
        sum_list = np.full((self.k, len(self.data[0])), 0.0)
        counter = np.full((self.k), 0)
        for day in range(len(self.data)):
            counter[int(self.centroid_choice[day])] += 1
            for day_val in range(len(self.data[day])):
                sum_list[int(self.centroid_choice[day])][day_val] += self.data[day][day_val]
        for i in range(len(sum_list)):
            for j in range(len(sum_list[i])):
                sum_list[i][j] = (sum_list[i][j] / counter[i])
        for i in range(len(self.centroids)):
            for j in range(len(self.centroids[i])):
                self.centroids[i][j] = sum_list[i][j]


    ##def find_min_max
    ##finds the minimal and maximum value of every day value and puts it in a array
    def find_min_max(self):
        for index in range(len(self.min_max)):
            self.min_max[index] = self.data[0]
        for day in self.data:
            for day_val in range(len(day)):
                if day[day_val] < self.min_max[0][day_val]:
                    self.min_max[0][day_val] = day[day_val]
                if day[day_val] > self.min_max[1][day_val]:
                    self.min_max[1][day_val] = day[day_val]

    ##def normalize
    ##day_val: Value that you want to check
    ##day_val_index: Index of which day value you want to normalize
    ##return: Normalized value between 0 and 1
    def normalize(self, day_val, day_val_index):
        return (day_val-self.min_max[0][day_val_index])/(self.min_max[1][day_val_index]-self.min_max[0][day_val_index])

    def normalize_data(self):
        for day_index in range(len(self.data)):
            for day_val_index in range(len(self.data[day_index])):
                self.data[day_index][day_val_index] = self.normalize(self.data[day_index][day_val_index], day_val_index)

    ##def pythagoras
    ##start_point: Value of the starting point
    ##end_point: Value of the end point
    ##Return: Return distance between 2 points
    ## Get the distance between 2 points
    def pythagoras(self, centroid, day_val):
        diff = 0
        for i in range(len(day_val)):
            diff += (day_val[i] - centroid[i]) ** 2
        return diff

    ##def screeplot
    ## Make a screeplot to see how good each K is
    def screeplot(self):
        plt.plot(self.all_K_used, self.all_shortest_dis, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('K')
        plt.ylabel('intra_distances')
        plt.show()


    ##def loop
    ##The loop for kMeans
    def loop(self):
        self.find_min_max()

        self.normalize_data()
        for k in range(1, 10):
            self.set_k(k)
            self.all_K_used = np.append(self.all_K_used, k)
            # for reset in range(5):
            self.empty_centroids()
            self.create_centroids()
            prev_centroids = copy.deepcopy(self.centroids)
            for looper in range(100):
                loop_again = False
                cur_centroids = copy.deepcopy(self.centroids)
                for day in range(len(self.data)):
                    self.centroid_choice[day] = self.closest_centroid(self.data[day])
                self.choose_new_centroid()
                if looper == 0:
                    continue
                for row in range(len(prev_centroids)):
                    for column in range(len(prev_centroids[row])):
                        if prev_centroids[row][column] != cur_centroids[row][column]:
                            prev_centroids = cur_centroids
                            loop_again = True
                            continue
                if not loop_again:
                    break
            self.all_shortest_dis = np.append(self.all_shortest_dis, self.count_distance())
        self.screeplot()

if __name__ == "__main__":
    kMeans_handler = kMeans(4, "C:/ti-software/AAAI/Opdrachten/3.1 kNN/dataset1.csv")
    kMeans_handler.loop()
