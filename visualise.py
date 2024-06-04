import numpy as np
import wandb
import matplotlib.pyplot as plt


def log_values_sorted(trained_with_flag, sorted_gradient_ind, training_size, portion, tag=""):
    poisoned = sum([x[-1] for x in trained_with_flag])

    if len(tag) > 0:
        tag = "_" + tag

    for t in range(10, training_size, 10):
        found = sum(trained_with_flag[sorted_gradient_ind[i][0]][2] for i in range(t)) # Selecting the corruption tag from the ordered list of indexes - which have been ordered according to the OT

        print(
            f"inspected: {t}, found: {found} detection rate: {found / poisoned:.2f} baseline: {t*portion}"
        )

        wandb.log(
            data = {
                "baseline": t*portion, 
                f"found{tag}": found,
                "custom_step": t,
            }, 
        )


# Visualize based on sorted values (calibrated gradient)
# Prints 3 graphs, with a random baselines (explained in paper...)
def visualize_values_distr_sorted(tdid, tsidx, trsize, portion, trainGradient):
    """
    tdid: list of [img, label, corrupted T/F]
    tsidx: list of indexes fo the gradients which are ordered by the magnitute of the gradietn fo the OT - largest magnitude first
    """
    x1, y1, base = [], [], []
    poisoned = trsize * portion
    for vari in range(10, trsize, 10):
        if vari < 3000:
            found = sum(tdid[tsidx[i][0]][2] for i in range(vari)) # Selecting the corruption tag from the ordered list of indexes - which have been ordered according to the OT

            #             print('inspected: '+str(vari), 'found: '+str(found),
            #                   'detection rate: ', str(found / poisoned), 'baseline = '+str(vari*0.2*0.9))

            print(
                f"inspected: {vari}, found: {found} detection rate: {found / poisoned:.2f} baseline: {vari*0.2*0.9}"
            )

        x1.append(vari)
        y1.append(sum(tdid[tsidx[i][0]][2] for i in range(vari)))
        base.append(vari * portion * 1.0)
    plt.scatter(x1, y1, s=10)
    plt.scatter(x1, base, s=10)
    # naming the x axis
    plt.xlabel("Inspected Images")
    # naming the y axis
    plt.ylabel("Detected Images")
    plt.yticks([0, 1])

    # giving a title to my graph
    plt.title("Detection vs Gradient Inspection")

    # function to show the plot
    plt.show()

    ################# GETTING POISON FLAG WITH GRADIENT ############
    x, y = [], []
    poison_cnt = 0
    last_ind = -1
    x_poisoned = []
    non_poisoned = []
    for i in range(trsize):
        x.append(trainGradient[i])
        # print(trainGradient[i])
        oriid = tsidx[i][0]
        y.append(tdid[oriid][2])
        poison_cnt += 1 if tdid[oriid][2] else 0
        last_ind = i if tdid[oriid][2] else last_ind
        if tdid[oriid][2]:
            x_poisoned.append(trainGradient[i])
        else:
            non_poisoned.append(trainGradient[i])
    plt.scatter(x, y, s=10)

    # naming the x axis
    plt.xlabel("Gradient")
    # naming the y axis
    plt.ylabel("Poisoned Image")
    plt.yticks([0, 1])

    # giving a title to my graph
    plt.title("Gradient vs Poisoned")

    # function to show the plot
    plt.show()

    print("Number of poisoned images: ", poison_cnt, " out of 10000.")
    print("last index of poison", last_ind)

    ########################### HISTOGRAM PLOT #################################################
    tminElement = np.amin(trainGradient)
    tmaxElement = np.amax(trainGradient)
    bins = np.linspace(tminElement, tmaxElement, 200)
    plt.hist(non_poisoned, bins, label="Clean Images")
    plt.hist(
        x_poisoned,
        bins,
        label="Poisoned Images",
        edgecolor="None",
        alpha=0.5,
    )
    # naming the x axis
    plt.xlabel("Gradient")
    # naming the y axis
    plt.ylabel("Number of Images")
    plt.title("Gradient of Poisoned and Non-Poisoned Images Lambda=(1,1)")
    plt.legend(loc="upper left")
    plt.show()


# Loading Baseline Data Values and Visualize
# To Be Implemented
def visualize_baselines():
    return
