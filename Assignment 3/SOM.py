# Self-Organized Map
import numpy as np
import scipy.spatial.distance as dist
import json


class SOM:

    # constructor
    def __init__(self, shape, number_of_feature,  samples_for_init,
                 distance_measure_str='cosine', topology='cubic',
                 init_learning_rate=0.01, max_epoch=200):
        """
        This is constructor of self-organized map.
        :param shape: shape of network, should be in form (a, b, c)
        :param number_of_feature: number of feature of input vector
        :param distance_measure_str: kind of distance measure, euclidean, manhattan,...
        :param topology: different topology for neighbours, line, square, Hexadecimal, circle
        :param init_learning_rate: initial learning rate
        :param max_epoch: maximum number of epochs that allowed to train network
        :param samples_for_init: it contains n=shape[0]*shape[1]*shape[2] different samples
                            for initializing SOM
        :return:
        """

        # number of features of input vector (number of weights of each neuron)
        self.num_weights_each_neuron = number_of_feature

        # shape of network, in form (row, col, depth)
        if len(shape) < 3:
            raise ValueError('Shape should be in form (a, b, c), like (2,3,1)')
        self.shape = np.copy(shape).tolist()

        # weights of neuron in networks
        #self.network = np.zeros(shape=(self.shape[0]*self.shape[1]*self.shape[2], self.num_weights_each_neuron))

        # initiate weights of network with samples
        if samples_for_init.shape[0] != (self.shape[0]*self.shape[1]*self.shape[2]):
            raise ValueError('Number of Neurons and initial weights aren\'t equal!')
        self.network = np.copy(samples_for_init)

        # distance function
        self.distance_fun_name = distance_measure_str
        self.distance_fun = None
        if distance_measure_str == 'euclidean':
            # euclidean distance
            self.distance_fun = dist.euclidean
        elif distance_measure_str == 'cityblock':
            # cityblock distance
            self.distance_fun = dist.cityblock
        elif distance_measure_str == 'chebyshev':
            # chebyshev distance
            self.distance_fun = dist.chebyshev
        elif distance_measure_str == 'cosine':
            # cosine distance
            self.distance_fun = dist.cosine
        else:
            raise ValueError('Name of distance type is incorrect!')

        # topology
        self.neighbours = None
        if topology == 'cubic':
            self.neighbours = self.topology_cubic
        elif topology == 'topology_hex':
            self.neighbours = self.topology_hex
        else:
            raise ValueError('Name of topology isn\'t correct')

        # maximum iteration that is allowed for training network
        self.max_epoch = max_epoch

        # initial learning rate
        self.initial_learning_rate = init_learning_rate

        # how many time does a neuron win a competition
        self.neuron_winning_all_epoch = {key:0 for key in range(0, self.shape[0]*self.shape[1]*self.shape[2])}
        self.neuron_winning_last_epoch = {key: 0 for key in range(0, self.shape[0] * self.shape[1] * self.shape[2])}

    def topology_cubic(self, indeces, r):
        """
        return indeces of neuron that are neighbour of a neuron that is located
        in networks_structure[indeces]. neighbourhood is a cube with radius r.
        in 1-D network it is line
        in 2-D network it is square
        in 3-d network it is cube
        :param indeces: in form of (a, b, c)
        :return: return indeces of neighbours in form
                        [[nb1-a, nb1-b, nb1-c],
                         [nb2-a, nb2-b, nb2-c],
                         ...,
                         [nbN-a, nbN-b, nbN-c]]
        """
        depth = (np.max((0, indeces[2]-r)), np.min((self.shape[2]-1, indeces[2] + r)))
        col = (np.max((0, indeces[1] - r)), np.min((self.shape[1]-1, indeces[1] + r)))
        row = (np.max((0, indeces[0] - r)), np.min((self.shape[0]-1, indeces[0] + r)))

        list_neighbours = []
        for k in range(depth[0], depth[1]+1):
            for i in range(row[0], row[1] + 1):
                for j in range(col[0], col[1] + 1):
                    list_neighbours.append([i, j, k])

        # return cubic indeces of the neuron
        return list_neighbours

    def topology_hex(self, indeces, r):
        """
        return indeces of neuron that are neighbour of a neuron that is located
        in networks_structure[indeces]. neighbourhood is a hex with radius r
        :param indeces: in form of (a, b, c)
        :return: return indeces of neighbours in form
                        [[nb1-a, nb1-b, nb1-c],
                         [nb2-a, nb2-b, nb2-c],
                         ...,
                         [nbN-a, nbN-b, nbN-c]]
        """
        depth = (np.max((0, indeces[2]-r)), np.min((self.shape[2]-1, indeces[2] + r)))
        col = (np.max((0, indeces[1] - r)), np.min((self.shape[1]-1, indeces[1] + r)))
        row = (np.max((0, indeces[0] - r)), np.min((self.shape[0]-1, indeces[0] + r)))

        list_neighbours = []
        for k in range(depth[0], depth[1]+1):
            for i in range(row[0], row[1] + 1):
                for j in range(col[0], col[1] + 1):
                    list_neighbours.append([i, j, k])

        # return hex indeces of the neuron
        return list_neighbours

    def unravel_index(self, index):
        """
        This function gets an index in one dimensional array and
        returns equivalent index in networks shape. network is
        1d, 2d or 3d matrix.
        :param index: index in one dimensional array
        :return: index in matrix is size= (network shape)
        """
        idx = np.unravel_index(index, self.shape)

        z = index // (self.shape[0] * self.shape[1])
        index -= (index // (self.shape[0] * self.shape[1])) * (self.shape[0] * self.shape[1])
        y = index % (self.shape[1])
        x = (index // (self.shape[1]))

        # return equivalent index in shape of network
        return x, y, z

    def ravel_index(self, indices):
        """
        This function get an index in form (a, b, c) then it returns
        equivalent position in 1d
        :param indices:
        :return:
        """
        return indices[0] * self.shape[1] + indices[1] + indices[2] * (self.shape[0]*self.shape[1])

    def learning_rate(self, learning_rate_0, previous_learning_rate, cur_iter, final_iter):
        """
        exponential learning rate decay
        :param learning_rate_0: initial learning rate
        :param previous_learning_rate: learning rate of previous iteration
        :param cur_iter: current iteration
        :param final_iter: final iteration
        :return:
        """
        min_rate = 0.01
        if previous_learning_rate <= min_rate:
            return min_rate
        else:
            learning_rate = learning_rate_0 * np.exp(-cur_iter/final_iter)
            if learning_rate >= min_rate:
                return learning_rate
            else:
                return min_rate

    def neighbourhood_size(self, nb_0, previous_nb, cur_iter, final_iter):
        """
        neighbourhood decay
        :param nb_0: initial radious of learning rate
        :param previous_nb: neighbourhood size of previous iteration
        :param cur_iter: current iteration
        :param final_iter: final iteration
        :return:
        """
        if previous_nb <= 0:
            return int(0)
        else:
            #return int(round(nb_0 * (1-cur_iter/final_iter)))
            return int(round(nb_0 * np.exp(-cur_iter / final_iter)))

    def neighbour_strength(self, std, index_winner, index_neighbour):
        """
        gaussian strength of neurons in neighbourhood of winner neuron
        :param std: standard deviation-equal size of neighbourhood
        :param index_winner: indexes of winner neuron in network
        :param index_neighbour: indexes of a neighbour to winner neuron in network
        :return: return strength, a scalar
        """
        if std == 0:
            std = 1

        # distance between winner neuron and neighbour
        dist_win_nei_2 = dist.euclidean(index_winner, index_neighbour)**2
        ns = np.exp(-dist_win_nei_2/(2*std*std))

        return ns

    def winner_neuron(self, x):
        """
        It computes distance of input sample to all neurons and returns index of
        winner neuron
        :param x: input sample
        :return: indexes of winner
        """
        dist_to_neurons = []
        for i in range(0, self.network.shape[0]):
            #print(self.network[i].shape, x.shape)
            #print(i, ': ', self.network[i], x, self.distance_fun(u=self.network[i], v=x))
            dist_to_neurons.append(self.distance_fun(u=self.network[i], v=x))
            #print(self.network[i])
            #print(x)
            #print(self.distance_fun(u=self.network[i], v=x))
            #print('-------------------^-----------------------')
        return np.argmin(dist_to_neurons)

    def fit(self, X):
        """
        compute weights of cluster for each neuron
        :param X: numpy.ndarray, each row is an observation
        :return:
        """
        previous_learning_rate = 1000
        previous_nb_size = 1000
        init_neighbourhood_radius = np.floor(np.max(self.shape)/2)

        # for maximum number of epoch,
        for t in range(0, self.max_epoch):
            # print current iteration
            print('Iter {}/{}'.format(t+1, self.max_epoch))

            # radius of neighbourhood
            radius = self.neighbourhood_size(nb_0=init_neighbourhood_radius,
                                             previous_nb=previous_nb_size, cur_iter=t+1,
                                             final_iter=self.max_epoch)
            previous_nb_size = radius

            # learning rate
            learning_rate = self.learning_rate(learning_rate_0=self.initial_learning_rate,
                                               previous_learning_rate=previous_learning_rate,
                                               cur_iter=t+1,
                                               final_iter=self.max_epoch)
            previous_learning_rate = learning_rate

            # reset number of winning in last epoch
            #for key in self.neuron_winning_last_epoch:
            #    self.neuron_winning_last_epoch[key] = 0

            # for all samples
            for i in range(0, X.shape[0]):

                # find winner neuron
                winner_1d_index = self.winner_neuron(x=X[i])
                winner_indexes = self.unravel_index(winner_1d_index)

                # count how many times a neuron wins
                #self.neuron_winning_all_epoch[winner_1d_index] += 1
                #self.neuron_winning_last_epoch[winner_1d_index] += 1

                # get neighbour of winner neuron
                neighours = self.neighbours(indeces=winner_indexes, r=radius)

                # update weights of winner neuron and
                # neurons that are in neighbourhood of winner
                for nb_indexes in neighours:
                    # strength decay of neighbour
                    strength = self.neighbour_strength(std=radius,
                                                       index_winner=winner_indexes,
                                                       index_neighbour=nb_indexes)
                    # convert index form (a,b,c) to 1-d index
                    nb_1d_index = self.ravel_index(indices=nb_indexes)

                    # update weights of neuron with respect to previous weight, learning rate
                    # strength of neighbour neuron in neighbourhood and difference between input and weights of the neuron

                    #print('lr: {}\nstr: {}\nwe{}'.format(learning_rate, strength, self.network[nb_1d_index]))
                    #print(X[i])
                    #print(self.network[nb_1d_index])
                    #print(X[i]-self.network[nb_1d_index])

                    self.network[nb_1d_index] = self.network[nb_1d_index] + learning_rate * strength * (X[i]-self.network[nb_1d_index])

            #print(learning_rate)
            #print(radius)
        self.write_in_file()
        #print('How many times a neuron won the competition in entire epochs:\n', self.neuron_winning_all_epoch)
        #print('How many times a neuron won the competition in last epochs:\n',self.neuron_winning_last_epoch)

    def predict(self, X):
        """
        compute index of winner neuron for each observation
        :param X:
        :return:
        """
        # hold cluster of each sample
        winners_1d_idx = []

        # for all samples
        for i in range(0, X.shape[0]):
            # find winner neuron
            winner_1d_index = self.winner_neuron(x=X[i])
            #winner_indexes = self.unravel_index(winner_1d_index)

            winners_1d_idx.append(winner_1d_index)

        winners_1d_idx = np.reshape(winners_1d_idx, (X.shape[0], 1))

        return winners_1d_idx

    def write_in_file(self):
        """
        Write trained SOM in file 'output.json'
        :return:
        """
        data = {}
        data['shape'] = self.shape
        data['weights'] = self.network.tolist()
        data['distance_fun'] = self.distance_fun_name
        with open('som_net.json', 'w') as outfile:
            json.dump(data, outfile)

    def load_from_file(self, file_name):
        """
        Load som from file
        :param file_name:
        :return:
        """
        with open(file_name) as data_file:
            data = json.load(data_file)

        self.shape = data['shape']
        self.network = np.copy(data['weights'])

        # distance function
        self.distance_fun_name = data['distance_fun']
        if self.distance_fun_name == 'euclidean':
            # euclidean distance
            self.distance_fun = dist.euclidean
        elif self.distance_fun_name == 'cityblock':
            # cityblock distance
            self.distance_fun = dist.cityblock
        elif self.distance_fun_name == 'chebyshev':
            # chebyshev distance
            self.distance_fun = dist.chebyshev
        elif self.distance_fun_name == 'cosine':
            # cosine distance
            self.distance_fun = dist.cosine
        else:
            raise ValueError('Name of distance type is incorrect!')


if __name__ == '__main__':

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from measures import purity_measure, rand_index
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from create_clean_dataset import create_clean_dataset
    #np.random.seed(0)

    # Read Iris dataset
    iris = datasets.load_iris()

    # Use features petal length, petal width
    X = iris.data
    y = iris.target

    #X, y, label_class = create_clean_dataset(max_features=2000)


    #X = np.array([[-50,20], [-25,1], [-30,15], [-30,25], [50,2], [50,-10], [40,15], [55,-10], [0,10], [10,0], [5,1], [-2,1], [-6,5], [4,3]])
    #y = np.array([[0],[0],[0],[0],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2]])

    # Split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    y_train = y_train.reshape((X_train.shape[0], 1))
    y_test = y_test.reshape((X_test.shape[0], 1))

    # standardization
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    ############################
    # SOM
    ############################
    print("###############################\n###########  SOM  #############\n###############################")
    # select some samples as random weights fo initialization
    initial_sample_indexes = np.random.permutation( X_train.shape[0])

    #shape_net = (4, 5, 1)
    shape_net = (2, 3, 2)
    number_of_neurons = shape_net[0] * shape_net[1] * shape_net[2]
    #som = SOM(shape=shape_net, number_of_feature=X_train.shape[1], distance_measure_str='euclidean',
    #          topology='cubic', init_learning_rate=0.1, max_epoch=300,
    #          samples_for_init=X_train[initial_sample_indexes[0:number_of_neurons]])

    som = SOM(shape=shape_net, number_of_feature=X_train.shape[1], distance_measure_str='cosine',
              topology='cubic', init_learning_rate=0.1, max_epoch=300,
              samples_for_init=X_train[initial_sample_indexes[0:number_of_neurons]])

    som.fit(X=X_train)
    #som.load_from_file('som_net.json')

    y_train_pre = som.predict(X=X_train)
    y_test_pre = som.predict(X=X_test)

    print('PU-Train-SOM: ', purity_measure(clusters=y_train_pre, classes=y_train))
    print('PU-Test-SOM: ', purity_measure(clusters=y_test_pre, classes=y_test))

    print('RI-Train-SOM: ', rand_index(clusters=y_train_pre, classes=y_train))
    print('RI-Test-SOM: ', rand_index(clusters=y_test_pre, classes=y_test))

    ############################
    # kmeans
    ############################
    print("###############################\n##########  Kmean  ############\n###############################")
    kmeans = KMeans(n_clusters=20, random_state=0).fit(X_train)
    y_train_pre = kmeans.predict(X_train)
    y_train_pre = y_train_pre.reshape(y_train_pre.shape[0],1)
    y_test_pre = kmeans.predict(X_test)
    y_test_pre = y_test_pre.reshape(y_test_pre.shape[0], 1)

    print('PU Train-Kmean: ', purity_measure(clusters=y_train_pre, classes=y_train))
    print('PU Test-Kmean: ', purity_measure(clusters=y_test_pre, classes=y_test))

    print('RI Train-Kmean: ', rand_index(clusters=y_train_pre, classes=y_train))
    print('RI Test-Kmean: ', rand_index(clusters=y_test_pre, classes=y_test))