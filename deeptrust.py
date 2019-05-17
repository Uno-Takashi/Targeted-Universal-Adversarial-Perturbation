import numpy as np

# We do not use this time!!!!!
# Calc maximum confidence image.

def deeptrust(image, f, grads,f_end,   max_iter=10,trust_target_rate=0.7 ,num_classes=2):

    """
       :param image: Image of size HxWx3
       :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param grads: gradient functions with respect to input (as many gradients as classes).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    f_image = np.array(f(image)).flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    num_classes=num_classes
    I = I[0:num_classes]
    label = I[0]

    input_shape = image.shape
    pert_image = image
    trust_rate=np.max(f_end(pert_image), axis=1).flatten()
    base_rate=trust_rate
                                          
    f_i = np.array(f(pert_image)).flatten()
    k_i = int(np.argmax(f_i))

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    while trust_rate<=trust_target_rate and loop_i < max_iter and abs(trust_rate-base_rate)<=0.2:

        pert = np.inf
        gradients = np.asarray(grads(pert_image,I))


            # set new w_k and new f_k
        w = gradients[0, :, :, :, :]
        f_t =  f_i[I[0]]
        pert = abs(f_t)/np.linalg.norm(w.flatten())

        # compute r_i and r_tot
        r_i =  pert * w / np.linalg.norm(w)
        r_tot = r_tot + r_i

        # compute new perturbed image
        pert_image = image + r_tot
        loop_i += 1

        # compute new label
        f_i = np.array(f(pert_image)).flatten()
        k_i = int(np.argmax(f_i))

        trust_rate=np.max(f_end(pert_image), axis=1).flatten()

    print(base_rate," --> ",trust_rate)

    return r_tot, loop_i, k_i, pert_image
