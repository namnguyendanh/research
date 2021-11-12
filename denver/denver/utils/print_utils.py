# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import logging
import itertools
import numpy as np

from tabulate import tabulate
from denver.functional import MaxProbCELoss

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except Exception as e:
    logger.debug(f"{e}")

def print_denver(message, denver_version):
    print("")
    print('\n'.join([
        '▅ ▆ ▇ █ Ⓓ ⓔ ⓝ ⓥ ⓔ ⓡ  █ ▇ ▆ ▅ {}'.format(denver_version), 
        ''
    ]))

def print_style_free(message, print_fun=print):
    print_fun("")
    print_fun("░▒▓█  {}".format(message))

def print_style_time(message, print_fun=print):
    print_fun("")
    print_fun("⏰  {}".format(message))
    print_fun("")
    
def print_style_warning(message, print_fun=print):
    print_fun("")
    print_fun("⛔️  {}".format(message))
    print_fun("")
    
def print_style_notice(message, print_fun=print):
    print_fun("")
    print_fun("📌  {}".format(message))
    print_fun("")

def print_line(text, print_fun=print):
    print_fun("")
    print_fun("➖➖➖➖➖➖➖➖➖➖ {} ➖➖➖➖➖➖➖➖➖➖".format(text.upper()))
    print_fun("")


def print_boxed(text, print_fun=print):
    box_width = len(text) + 2
    print_fun('')
    print_fun('╒{}╕'.format('═' * box_width))
    print_fun('  {}  '.format(text.upper()))
    print_fun('╘{}╛'.format('═' * box_width))
    print_fun('')

def view_table(metrics):
    """Function view table logger.

    :param metrics: A dict type. FORMAT: {'key1': [a, b, c], 'key2': [d, e, f]}
    """
    if "tag_detailed_results" in metrics:
        scores, by_classes = get_detail_results(metrics['tag_detailed_results'][0])
        _ = metrics.pop("tag_detailed_results", None)
        _ = metrics.update(scores)

        print(tabulate(metrics, headers="keys", tablefmt='pretty'))
        print("Detailed results: ")
        print(tabulate(by_classes, headers="keys", tablefmt='pretty'))
    
    elif "cls_detailed_results" in metrics:
        cls_detailed = metrics.get("cls_detailed_results")
        _ = metrics.pop("cls_detailed_results", None)

        print(tabulate(metrics, headers="keys", tablefmt='pretty'))
        if cls_detailed:
            print("Detailed results: ")
            print(cls_detailed)

    elif "intent" in metrics or "tags" in metrics:
        imetrics = metrics.get("intent")
        if imetrics:
            _ = metrics.pop("intent", None)

        
        cls_detailed = metrics.get("cls_detailed")
        if cls_detailed:
            _ = metrics.pop("cls_detailed", None)

        tmetrics = metrics.get("tags")
        if tmetrics:
            _ = metrics.pop("tags", None)
        
        tags_detailed = metrics.get("tags_detailed")
        if tags_detailed:
            _ = metrics.pop("tags_detailed", None)

        print(tabulate(metrics, headers="keys", tablefmt='pretty'))

        if imetrics:
            print("Intent results: ")
            print(tabulate(imetrics, headers="keys", tablefmt='pretty'))
        if cls_detailed:
            print("Intent detailed results: ")
            print(cls_detailed)

        if tmetrics:
            print("Tags results: ")
            print(tabulate(tmetrics, headers="keys", tablefmt='pretty'))
        if tags_detailed:
            print("Tags detailed results: ")
            print(tags_detailed)

    else:
        print(tabulate(metrics, headers="keys", tablefmt='pretty'))

def to_list(string):
    values = [t.strip() for t in string.split(' - ')]
    tag = values[0].split()[0]
    tp = " ".join(values[0].split()[1:])
    values.insert(0, "tag: " + tag)
    values[1] = tp

    return values

def get_detail_results(detail_results):
    lines = detail_results.split('\n')

    scores = [lines[2].strip('- '), lines[3].strip('- ')]
    scores = {
        'f1-score (micro)': [scores[0].split()[-1].strip()],
        'f1-score (macro)': [scores[1].split()[-1].strip()]
    }

    
    tags_results = [to_list(lines[i].strip()) for i in range(6, len(lines))]

    by_classes = {
        'tag': [],
        'tp': [],
        'fp': [],
        'fn': [],
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    for tag_result in tags_results:
        for element in tag_result:
            temp = element.split(":")
            by_classes[temp[0].strip()].append(temp[1].strip())

    return scores, by_classes

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, 
                          save_dir=None):


    """Function to plot confusion matrics.

    :param cm: confusion_matrix: function in sklearn.
    :param target_names: list of classes.
    :param cmap: str or matplotlib Colormap: Colormap recognized by matplotlib.
    :param normalize: normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
    :param save_dir: str: directory address to save.
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    width = int(10/7*len(target_names))
    height = int(8/7*len(target_names))

    plt.figure(figsize=(width, height))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label. Metrics: accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        logger.debug(f"Save confusion-matrix...")
        plt.savefig((save_dir + '/{}.png'.format(title)))
    except IOError:
        logger.error(f"Could not save file in directory: {save_dir}")

class ConfidenceHistogram(MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)
        #scale each datapoint
        n = len(labels)
        w = np.ones(n) / n

        plt.rcParams["font.family"] = "serif"
        #size and axis limits 
        plt.figure(figsize=(8, 8))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0)    
        #plot histogram
        plt.hist(self.confidences, n_bins, weights = w,color='b', range=(0.0, 1.0), edgecolor = 'k')

        #plot vertical dashed lines
        acc = np.mean(self.accuracies)
        conf = np.mean(self.confidences)                
        plt.axvline(x=acc, color='tab:grey', linestyle='--', linewidth = 3)
        plt.axvline(x=conf, color='tab:grey', linestyle='--', linewidth = 3)
        if acc > conf:
            plt.text(acc + 0.03, 0.9, 'Accuracy', rotation=90, fontsize=11)
            plt.text(conf - 0.07, 0.9, 'Avg. Confidence', rotation=90, fontsize=11)
        else:
            plt.text(acc - 0.07, 0.9, 'Accuracy', rotation=90,fontsize=11)
            plt.text(conf + 0.03, 0.9, 'Avg. Confidence', rotation=90, fontsize=11)

        plt.ylabel('% of Samples', fontsize=13)
        plt.xlabel('Confidence', fontsize=13)
        plt.tight_layout()
        
        if title is not None:
            plt.title(title, fontsize=16)

        return plt

class ReliabilityDiagram(MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        #computations
        delta = 1.0/n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,n_bins)
        error = np.abs(np.subtract(mid,self.bin_acc))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(8, 8))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0)
        #plot bars and identity line
        plt.bar(
            x, self.bin_acc, color = 'b', width=delta,align='edge',
            edgecolor = 'k',label='Outputs',zorder=5
        )
        plt.bar(
            x, error, bottom=np.minimum(self.bin_acc,mid), color = 'mistyrose', 
            alpha=0.5, width=delta, align='edge', edgecolor = 'r', hatch='/', label='Gap', zorder=10
        )
        ident = [0.0, 1.0]
        plt.plot(ident, ident, linestyle='--', color='tab:grey', zorder=15)
        #labels and legend
        plt.ylabel('Accuracy', fontsize=13)
        plt.xlabel('Confidence', fontsize=13)
        plt.legend(loc='upper left', framealpha=1.0, fontsize='medium')

        if title is not None:
            plt.title(title, fontsize=16)
        plt.tight_layout()

        return plt