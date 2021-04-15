import torch
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def plot_2D_labeled_data(X,y,fig_number,fig_title):
    # put plt.ioff() and plt.show() at end 
    plt.ion()
    f = plt.figure(fig_number)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.axis('equal')
    plt.title(fig_title)
    f.show()

# Choosing `num_centers` random data points as the initial centers
def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers

# Compute for each data point the closest center
def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long, device=device)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes

# Compute new centers as means of the data points forming the clusters
def update_centers(dataset, codes, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device))
    centers /= cnt.view(-1, 1)
    return centers

def cluster(dataset, num_centers):
    centers = random_init(dataset, num_centers)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            sys.stdout.write('\n')
            print('Converged in %d iterations' % num_iterations)
            break
        codes = new_codes
    return centers, codes

if __name__ == '__main__':
    n = 1000
    d = 100
    num_centers = 3
    # It's (much) better to use 32-bit floats, for the sake of performance

    mean1 = [-1,-1]
    mean2 = [1,1]
    cov = [[0.95,0.05],[0.05,0.95]]

    num_points_per = 500

    # generate data 
    d1 = np.random.multivariate_normal(mean1, cov, num_points_per)
    d2 = np.random.multivariate_normal(mean2, cov, num_points_per)

    init_labels = np.append(np.zeros((1,num_points_per)),np.zeros((1,num_points_per))+1)

    dataset_numpy = np.append(d1,d1,axis=0).astype(np.float32)
    dataset = torch.from_numpy(dataset_numpy).to(device)
    print('Starting clustering')
    centers, codes = cluster(dataset, num_centers)

    plot_2D_labeled_data(dataset,init_labels,1,"test data")
    plot_2D_labeled_data(dataset,codes,2,"labeled data")

    plt.ioff()
    plt.show()

    print(codes)
