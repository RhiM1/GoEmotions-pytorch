from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import torch
# from exemplar.data.Datasets.datasets import timit_hubert_dataset
# from minerva2 import minerva3_phoneRepLearn, minerva3_vowelRep, minerva3_vowelRepLearnEx
# from datasets_vowels import timit_context_dataset_vowels
from model import minerva_ffnn2
import argparse
from w2v_dataset import get_goem_dataset
from sentence_transformers import SentenceTransformer, util
from constants import DATAROOT_group, DATAROOT_ekman, DATAROOT_original


def plot_class_and_ex_reps(exReps, exClasses, model_reps, classOrder=None, classOrderEx=None, ID_to_class=None, classReps = None, num_classes = 4):
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
        8: 'positive',
        2: 'negative',
        4: 'neutral',
        1: 'ambiguous',
        10: '+ve/-ve',
        3: 'ambi/-ve',
        5: 'ambi/neut',
        6: '-ve/neut',
        7: 'ambi/-ve/neu',
        9: 'amb/+ve',
        11: 'amb/-ve/+ve',
        12: 'neut/+ve',
        13: 'amb/neut/+ve',
        14: '-ve/neut/+ve',
        15: 'amb/-ve/neut/+ve'

    }
    class_order = [8, 2, 4, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]

    order = [torch.arange(len(exReps))[ex_checked_classes == i] for i in class_order]
    order = torch.cat(order).tolist()
    # order = []
    # for orders in orderss:
    #     order.append(orders)
    print(f"order:\n{order}")
    exReps = exReps[order]
    ex_checked_classes = ex_checked_classes[order]
    
    if classOrderEx is not None:
        print(np.shape(phoneReps))
        phoneReps = phoneReps[classOrderEx, :]
        print(np.shape(phoneReps))
    if classOrder is not None:
        phoneReps = phoneReps[:, classOrder]
        print(np.shape(phoneReps))

    plt.figure(figsize=(3.5,2))

    ax = sb.scatterplot(x = exReps[:, 0], y = exReps[:, 1], hue = [id_to_class[int(classID)] for classID in ex_checked_classes.tolist()]) #, annot=False, cmap=sb.color_palette("rocket_r", as_cmap=True), cbar_kws={'label': 'Scale'})
    ax.set(ylabel="Rep 2", xlabel="Rep 1")
    ax.plot(model_reps[0], model_reps[1], color = 'black', marker = 'x', markersize = 14)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.05, 1.05)
    # for i, model_rep in enumerate(model_reps):
    #     d = model_rep - exReps[i]
    #     ax.arrow(exReps[i, 0], exReps[i, 1], d[0], d[1])

    for classID in range(num_classes):
        plt.text(x = -1, y = 0, s = 'negative')
        plt.text(x = 0.3, y = 0, s = 'positive')
        plt.text(x = -0.3, y = -0.9, s = 'neutral')
        plt.text(x = -0.4, y = 0.9, s = 'ambiguous')
        # plt.text(x = classReps[classID, 0], y = classReps[classID, 1], s = id_to_class[true_class_ids[classID]])

    # plt.legend([key for key in id_to_class.keys()], [value for value in id_to_class.values()])
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)
    leg.set_bbox_to_anchor((1.0, 1.25))
    # ax.legend(bbox_to_anchor=(1, 1.25))

    plt.show()

def track_class_and_ex_reps(epochs, exReps, exClasses, model_reps, classOrder=None, classOrderEx=None, ID_to_class=None, classReps = None, num_classes = 4):
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
        # for i, model_rep in enumerate(model_reps[plot_id]):
        #     d = exReps[plot_id][i] - model_rep
        #     ax.arrow(exReps[plot_id][i, 0], exReps[plot_id][i, 1], d[0], d[1])
        for classID in range(num_classes):
            axs[plot_id // 2, plot_id % 2].text(x = classReps[plot_id][classID, 0], y = classReps[plot_id][classID, 1], s = id_to_class[true_class_ids[classID]])
        if plot_id != 0:
             axs[plot_id // 2, plot_id % 2].get_legend().remove()
    plt.show()



# ------------------------------------------------------

def main(args):

    # epochs = [0, 5, 10, 15, 20, 25]
    epochs = None

    ID_to_emotion = {
        0: 'ambiguous',
        1: 'negative',
        2: 'neutral',
        3: 'positive'
    }
    
    args.model_name_or_path = 'paraphrase-mpnet-base-v2'
    feats_model = SentenceTransformer(args.model_name_or_path)
    test_dataset, test_texts = get_goem_dataset(args, mode = 'test', model = feats_model)
    train_dataset, train_texts = get_goem_dataset(args, mode = 'train', model = feats_model)
    batch = test_dataset[0:20]
    # print(texts)
    # print(batch[0])
    text_of_interest = 3
    print(batch[1])
    # print(test_texts[text_of_interest])
    # quit()

    # model = minerva_detEx(args, load_dir=args.trained_model_folder)
    model = minerva_ffnn2(args, load_dir=args.trained_model_folder)
    model.to('cpu')
    ex_classes = model.ex_classes
    # print(model.mult)
    # print(model.thresh)
    # _, _, model_reps = model(model.ex_features)
    if epochs is None:
        # model_reps = model(model.ex_features)
        # model_reps = model_reps['echo']
        output = model(batch[0])
        model_reps = output['echo']
        activations = output['activations']
        model_reps = torch.nn.functional.normalize(model_reps, dim = -1)
        # print(model_reps)
        # print(activations[text_of_interest].argmax())
        # print(model.ex_idx[activations[3].argmax()])
        print(test_texts[text_of_interest])
        # print(len(train_texts))
        print(train_texts[model.ex_idx[activations[text_of_interest].argmax()]])
        # print(model.ex_idx)
        # quit()
        ex_reps = model.ex_class_reps
        ex_reps = torch.nn.functional.normalize(ex_reps, dim = -1)

        # rand_perm = torch.randperm(len(model_reps))[0:100]
        # ex_reps = ex_reps[rand_perm]
        # model_reps = model_reps[rand_perm]
        # ex_classes = ex_classes[rand_perm]

        plot_class_and_ex_reps(
            exReps = ex_reps.to('cpu').detach(), 
            exClasses = ex_classes, 
            ID_to_class = ID_to_emotion, 
            classReps = model.class_reps.to('cpu').detach(),
            model_reps = model_reps[text_of_interest].to('cpu').detach()
        )
    else:
        ex_reps = []
        class_reps = []
        model_reps = []
        for epoch in epochs:
            model.load_pretrained(args.tracked_model_folder, epoch)
            model.to('cpu')
            ex_reps.append(model.ex_class_reps.to('cpu').detach())
            class_reps.append(model.class_reps.to('cpu').detach())
            _, _, mr = model(model.ex_features)
            model_reps.append(mr.to('cpu').detach())
        track_class_and_ex_reps(
            epochs = epochs,
            exReps = ex_reps,
            exClasses = ex_classes,
            ID_to_class = ID_to_emotion,
            classReps = class_reps,
            model_reps = model_reps
        )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model", help="phone classification model: ffnn, minerva2, minerva3", default="minerva"
    )
    parser.add_argument(
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_001_42"
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_002_42"
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_003_42"
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_004_42"
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_005_42"
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_006_42"
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_007_42"
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_008_42"
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_009_42"
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_010_42"
        # "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_013_42"
        "--model_folder", help="trained model folder name", default = "sen_trans_minerva_ffnn2_005_013_84"
    )

    args = parser.parse_args()

    args.trained_model_folder = f"ckpt/csl_paper/{args.model_folder}/checkpoint"
    args.tracked_model_folder = f"ckpt/csl_paper/{args.model_folder}/track_reps"
    args.data_dir = "data/group"
    args.feats_type = 'sen_trans'

    main(args)