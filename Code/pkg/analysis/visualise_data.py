import matplotlib.pyplot as plt
import numpy as np

def compute_differences(sample1: np.ndarray, sample2: np.ndarray) -> np.ndarray:
    output = sample1 - sample2
    return np.absolute(output)


def visualise_sentiment_differences(dataset1: np.ndarray, dataset2: np.ndarray):
    """
    Visualise the difference in sentiment between sarcastic and non-sarcastic instances in each class
    Larger bars show that the proportion of tokens labelled with that particular sentiment class differs more
    between sarcastic and non-sarcastic instances
    :param dataset1:
    :param dataset2:
    :return:
    """
    labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

    index = 0.6 * np.arange(len(labels))
    bar_width = 0.2

    plt.bar(index, dataset1, bar_width,
            alpha=1,
            color='dimgrey',
            label='Sarcasm Amazon Review Corpus')

    plt.bar(index + bar_width, dataset2, bar_width,
            alpha=0.7,
            color='dimgrey',
            label='News Headlines for Sarcasm Detection')

    plt.ylabel('Absolute differences between sarcastic\n and non-sarcastic instances', fontsize=13)
    plt.xlabel('Sentiment Labels', fontsize=13)
    plt.xticks(index + bar_width, labels)
    plt.legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig('Images/absolute_differences.png')


def visualise_sentiment(positive: np.ndarray, negative: np.ndarray):
    labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

    index = 0.6 * np.arange(len(labels))
    bar_width = 0.2

    plt.bar(index, positive, bar_width,
            alpha=1,
            color='dimgrey',
            label='Non-Sarcastic')

    plt.bar(index + bar_width, negative, bar_width,
            alpha=0.7,
            color='dimgrey',
            label='Sarcastic')

    plt.ylabel('Proportion of Tokens', fontsize=13)
    plt.xlabel('Sentiment Labels', fontsize=13)
    plt.xticks(index + bar_width, labels)
    plt.legend()
    # plt.title('Sarcasm Amazon Review Corpus', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()
    # plt.savefig('Images/SARC_sentimentlabels.png')


if __name__ == "__main__":
    sarc_pos = np.array(
        [0.005091248953004676, 0.041350809727765, 0.8994157796925195, 0.04229651150485239, 0.011845650121857931,
         1.607662140849572])
    sarc_neg = np.array(
        [0.0023434613237710553, 0.023207982123672997, 0.8944306116972885, 0.052378647503276816,
         0.027639297351990763, 1.9866354409562172])
    nhsd_pos = np.array(
        [0.0018528285173080475, 0.03934499539017465, 0.8955364365141866, 0.05614412600544106,
         0.007121613572856909, 1.595080925138135])
    nhsd_neg = np.array(
        [0.0016924451656707147, 0.030212075099074087, 0.907942537511219, 0.051825273876584095,
         0.008327668347401977, 1.6702002002002005])

    sarc_pos = sarc_pos[: -1]
    sarc_neg = sarc_neg[: -1]
    nhsd_pos = nhsd_pos[: -1]
    nhsd_neg = nhsd_neg[: -1]

    sarc_output = compute_differences(sarc_pos, sarc_neg)
    nhsd_output = compute_differences(nhsd_pos, nhsd_neg)

    # visualise_sentiment_differences(sarc_output, nhsd_output)

    visualise_sentiment(nhsd_pos, nhsd_neg)
    visualise_sentiment(sarc_pos, sarc_neg)