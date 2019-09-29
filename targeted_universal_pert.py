import numpy as np
from deeptarget import deeptarget
from util_univ import *

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def targeted_perturbation(dataset, f, grads,target, delta=0.2, max_iter_uni = np.inf, xi=10, p=np.inf, overshoot=0.02, max_iter_df=20):
    """
    :param dataset: Images of size MxHxWxC (M: number of images). I Recommend M > 5000

    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).

    :param grads: gradient functions with respect to input (as many gradients as classes).

    :param delta: controls the desired target fooling rate (default = 80% fooling rate)

    :target : target classes namber. 

    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)

    :param xi: controls the l_p magnitude of the perturbation (default = 10)

    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)

    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).

    :param max_iter_df: maximum number of iterations for deepfool (default = 10)

    :return: the universal perturbation.
    """

    v = 0
    target_fooling_rate = 0.0
    num_images =  int(np.shape(dataset)[0]) # The images should be stacked ALONG FIRST DIMENSION

    itr = 0
    while target_fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
        np.random.shuffle(dataset)

        print ('Starting pass number ', itr,'Target is ' , target)
        # Go through the data set and compute the perturbation increments sequentially
        for k in range(0, num_images):
            cur_img = dataset[k:(k+1), :, :, :]

            if int(np.argmax(np.array(f(cur_img+v)).flatten())) != int(target):
                
                print("\rProgress : ["+"#"*int(k/int(num_images/20))+"-"*(20-int(k/int(num_images/20)))+"] ", str(k).zfill(len(str(num_images))), ' / ',num_images,"," ,end="")
                # Compute adversarial perturbation
                dr,iter,pert_label,_ = deeptarget(cur_img + v, f, grads, overshoot=overshoot, max_iter=max_iter_df,target=target)

                
                print(" Tracking labels :",int(np.argmax(np.array(f(cur_img+v)).flatten())),"->",pert_label," "*6,end="")

                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr

                    # Project on l_p ball
                    v = proj_lp(v, xi, p)

        itr = itr + 1


        # Compute the target fooling rate
        target_fooling_rate = target_fooling_rate_calc(v=v,dataset=dataset,f=f,target=target)
        print("")
        print('TARGET FOOLING RATE = ', target_fooling_rate)

    return v