from sklearn.utils import shuffle

class KNeighborsClassifier():
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X=None
        self.y=None
        self.y_list=None
        self.dimension=1

    def fit(self, X, y):
        if not isinstance(X, list) or not isinstance(y, list):
            raise ValueError('Input type has to be list')
        self.X, self.y = shuffle(X, y, random_state=0)
        self.y_list=list(set(y))
        self.dimension = len(self.X[0])

    def predict(self, X):
        pred_results = []
        for target_x in X:
            predict=''
            dists=[]
            neighbors=[]
            labels = list(self.y)

            for i in range(len(self.X)):
                dist = 0
                for j in range(self.dimension):
                    dist += (self.X[i][j] - target_x[j])**2
                dists.append(dist)

            for i in range(self.n_neighbors):
                neighbors.append(self.__get_min(dists, labels))

            last_min_x = neighbors[self.n_neighbors-1][0]
            while last_min_x == min(dists):
                neighbors.append(self.__get_min(dists, labels))

            neighbor_labels = [i[1] for i in neighbors]
            max_cnt=0
            for label in self.y_list:
                cnt = neighbor_labels.count(label)
                if max_cnt < cnt :
                    max_cnt = cnt
                    predict = label

            pred_results.append(predict)
        if len(pred_results)==1 :
            return pred_results[0]
        return pred_results

    def __get_min(self, dists, labels):
        min_x = min(dists)
        min_index = dists.index(min_x)
        result = list([min_x, labels[min_index]])
        dists.remove(min_x)
        del labels[min_index]
        return result