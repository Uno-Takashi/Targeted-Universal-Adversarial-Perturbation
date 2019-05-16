import os
import numpy as np

def target_fooling_rate_calc(v,dataset,f,target):
    dataset_perturbed = dataset + v
    num_images =  np.shape(dataset)[0]
    est_labels_pert = np.zeros((num_images))

    batch_size = 100
    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

    # Compute the fooling rate
    target_fooling_rate = float(np.sum(int(est_labels_pert) == target) / float(num_images))
    return fooling_rate



def visualization_pert(v):
    plt.imshow(v)
    plt.imshow()


def class2label_str(num_pert):
    labels = open(os.path.join('data', 'labels.txt'), 'r').read().split('\n')
    return labels[np.int(num_pert)-1].split(',')[0]
