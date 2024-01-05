from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import torch
# from exemplar.data.Datasets.datasets import timit_hubert_dataset
# from minerva2 import minerva3_phoneRepLearn, minerva3_vowelRep, minerva3_vowelRepLearnEx
# from datasets_vowels import timit_context_dataset_vowels
from model import minerva_detEx
import argparse


def plot_class_and_ex_reps(exReps, exClasses, classOrder=None, classOrderEx=None, ID_to_class=None, classReps = None, num_classes = 4):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """

    ex_checked_classes = exClasses.to(torch.float) @ torch.tensor([1, 2, 4, 8]).to(torch.float)
    # print(ex_checked_classes)
    # print(ex_checked_classes.size())
    # for i in range(len(ex_checked_classes)):
    #     print(ex_checked_classes[i], exClasses[i])
    
    id_to_class = {
        1: 'ambiguous',
        2: '-ve',
        3: 'ambi/-ve',
        4: 'neutral',
        5: 'amb/neutral',
        6: '-ve/neutral',
        7: 'amb/-ve/neutral',
        8: '+ve',
        9: 'amb/+ve',
        10: '-ve/+ve',
        11: 'amb/-ve/+ve',
        12: 'neutral/+ve',
        13: 'amb/neutral/+ve',
        14: '-ve/neutral/+ve',
        15: 'amb/-ve/neutral/+ve'

    }
    
    if classOrderEx is not None:
        print(np.shape(phoneReps))
        phoneReps = phoneReps[classOrderEx, :]
        print(np.shape(phoneReps))
    if classOrder is not None:
        phoneReps = phoneReps[:, classOrder]
        print(np.shape(phoneReps))

    ax = sb.scatterplot(x = -exReps[:, 0], y = -exReps[:, 1], hue = [id_to_class[int(classID)] for classID in ex_checked_classes.tolist()]) #, annot=False, cmap=sb.color_palette("rocket_r", as_cmap=True), cbar_kws={'label': 'Scale'})
    ax.set(ylabel="Rep 2", xlabel="Rep 1")

    true_class_ids = [1, 2, 4, 8]
    for classID in range(num_classes):
        plt.text(x = -classReps[classID, 0], y = -classReps[classID, 1], s = id_to_class[true_class_ids[classID]])

 
    plt.show()

def track_class_and_ex_reps(epochs, exReps, exClasses, classOrder=None, classOrderEx=None, ID_to_class=None, classReps = None, num_classes = 4):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """

    ex_checked_classes = exClasses.to(torch.float) @ torch.tensor([1, 2, 4, 8]).to(torch.float)
    # print(ex_checked_classes)
    # print(ex_checked_classes.size())
    for i in range(len(ex_checked_classes)):
        print(ex_checked_classes[i], exClasses[i])
    
    id_to_class = {
        1: 'ambiguous',
        2: '-ve',
        3: 'ambi/-ve',
        4: 'neutral',
        5: 'amb/neutral',
        6: '-ve/neutral',
        7: 'amb/-ve/neutral',
        8: '+ve',
        9: 'amb/+ve',
        10: '-ve/+ve',
        11: 'amb/-ve/+ve',
        12: 'neutral/+ve',
        13: 'amb/neutral/+ve',
        14: '-ve/neutral/+ve',
        15: 'amb/-ve/neutral/+ve'
    }
    
    true_class_ids = [1, 2, 4, 8]

    # if classOrderEx is not None:
    #     print(np.shape(phoneReps[0]))
    #     phoneReps = phoneReps[classOrderEx, :]
    #     print(np.shape(phoneReps))
    # if classOrder is not None:
    #     phoneReps = phoneReps[:, classOrder]
    #     print(np.shape(phoneReps))


    fig, axs = plt.subplots(len(epochs) // 2, 2, sharex = True, sharey = True)
    for ax in axs.flat:
        ax.set(ylabel="Rep 2", xlabel="Rep 1")
    for ax in axs.flat:
        ax.label_outer()
    # axs[0, 0] = sb.scatterplot(x = exReps[0][:, 0], y = exReps[0][:, 1], hue = [id_to_class[int(classID)] for classID in ex_checked_classes.tolist()])
    # axs[0, 1] = sb.scatterplot(x = exReps[1][:, 0], y = exReps[1][:, 1], hue = [id_to_class[int(classID)] for classID in ex_checked_classes.tolist()])
    # axs[1, 0] = sb.scatterplot(x = exReps[2][:, 0], y = exReps[2][:, 1], hue = [id_to_class[int(classID)] for classID in ex_checked_classes.tolist()])
    # axs[1, 1] = sb.scatterplot(x = exReps[3][:, 0], y = exReps[3][:, 1], hue = [id_to_class[int(classID)] for classID in ex_checked_classes.tolist()])
    for plot_id in range(len(epochs)):
        axs[plot_id // 2, plot_id % 2].set_title(f"axis [{plot_id // 2}, {plot_id % 2}]")
        sb.scatterplot(x = exReps[plot_id][:, 0], y = exReps[plot_id][:, 1], hue = [id_to_class[int(classID)] for classID in ex_checked_classes.tolist()], ax = axs[plot_id // 2, plot_id % 2])
        # exit()
        # axs[plot_id // 2, plot_id % 2].set_title(f"epoch {epochs[plot_id]}")
        for classID in range(num_classes):
            axs[plot_id // 2, plot_id % 2].text(x = classReps[plot_id][classID, 0], y = classReps[plot_id][classID, 1], s = id_to_class[true_class_ids[classID]])
        if plot_id != 0:
             axs[plot_id // 2, plot_id % 2].get_legend().remove()
    plt.show()



# ------------------------------------------------------

def main(args):

    epochs = [0, 25, 50, 75, 100, 125]
    # epochs = None

    ID_to_emotion = {
        0: 'emotion 0',
        1: 'emotion 1',
        2: 'emotion 2',
        3: 'emotion 3'
    }

    model = minerva_detEx(args, load_dir=args.trained_model_folder)
    model.to('cpu')
    ex_classes = model.ex_classes
    if epochs is None:
        plot_class_and_ex_reps(
            exReps = model.ex_class_reps.to('cpu').detach(), 
            exClasses = ex_classes, 
            ID_to_class = ID_to_emotion, 
            classReps = model.class_reps.to('cpu').detach()
        )
    else:
        ex_reps = []
        class_reps = []
        for epoch in epochs:
            model.load_pretrained(args.tracked_model_folder, epoch)
            model.to('cpu')
            ex_reps.append(model.ex_class_reps.to('cpu').detach())
            class_reps.append(model.class_reps.to('cpu').detach())
        track_class_and_ex_reps(
            epochs = epochs,
            exReps = ex_reps,
            exClasses = ex_classes,
            ID_to_class = ID_to_emotion,
            classReps = class_reps
        )


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "--model", help="phone classification model: ffnn, minerva2, minerva3", default="minerva"
    )
    parser.add_argument(
        "--model_folder", help="trained model folder name", default = "sen_trans_minerva_detEx_007_004a_42"
    )

    args = parser.parse_args()

    args.trained_model_folder = f"ckpt/csl_paper/{args.model_folder}/checkpoint"
    args.tracked_model_folder = f"ckpt/csl_paper/{args.model_folder}/track_reps"

    main(args)