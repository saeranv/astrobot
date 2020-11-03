"""
Helpful tricks in jupyter

# reload modules
%load_ext autoreload
%autoreload 2

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Set precision
pd.set_option('precision', 2)
np.set_printoptions(precision=2)
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _translate_train_sizes
from sklearn.metrics import f1_score


IBLDTYPES = ['buildings', 'denseresidential',
             'mediumresidential', 'sparseresidential', 'mobilehomepark']

FEAT_LABELS = [('shgc', 'infil'), ('shgc', 'msys'), ('heateff', 'coolcap')]
FEAT_LABELS_IDX = [(0, 1), (0, 6), (4, 5)]


def pp(x, *args):
    pprint(x) if not args else print(x, *args)


def fd(module, key=None):
    """ To efficiently search modules in osm"""
    def hfd(m, k): return k.lower() in m.lower()
    if key is None:
        return [m for m in dir(module)][::-1]
    else:
        return [m for m in dir(module) if hfd(m, key)][::-1]


def bench_k_means(estimator, name, data, labels, sample_size):
    """Benchmark k-means clustering

    #sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
    This code is referenced and slightly modified from:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
    """

    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


def retrieve_info(_kmeans, cluster_labels, y_train):
    """
    Associates most probable label with each cluster in KMeans model
    This function is referenced from:
    https://medium.com/@joel_34096/k-means-clustering-for-image-classification-a648f28bdc47
    Returns:
        y prediction
    """
    label_dict = {}
    num_labels = len(np.unique(_kmeans.labels_))
    for i in range(num_labels):
        # boolean array of label to i
        idx = np.where(cluster_labels == i, 1, 0)
        # Match boolean distribution with ytrain
        num = np.bincount(y_train[idx == 1])
        if num.size == 0:
            break
        num = num.argmax()
        label_dict[i] = num
    if len(label_dict.keys()) < num_labels:
        return False
    return [label_dict[label] for label in _kmeans.labels_ if label in label_dict], label_dict


def plot_clusters(km3, reduced_data, ytrain, ax, title_txt, feats='', xlabel=None, ylabel=None):
    """Plot clustering plots"""
    km3.fit(reduced_data)
    h = .02
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = km3.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.figure(1)
    # plt.clf()
    ax.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap=plt.cm.Paired,
              aspect='auto', origin='lower')

    ax.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = km3.cluster_centers_
    cols = ['blue', 'purple', 'red']  # 0, 1, 2
    for centroid, col in zip(centroids, cols):
        ax.scatter(centroid[0], centroid[1],
                   marker='x', s=169, linewidths=3,
                   color=col, zorder=10)

    ypred = retrieve_info(km3, km3.labels_, ytrain)
    f1 = metrics.f1_score(ypred, ytrain, average='macro')

    # Plot text
    xfeat, yfeat = feats
    title_txt = title_txt + ' ' + xfeat + ' vs ' + yfeat
    title_txt = 'K-means {}\n'.format(title_txt)

    ax.set_title(title_txt + 'f1: {}'.format(f1))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_xticks(())
    # ax.set_yticks(())
    if xlabel is not None:
        xfeat = xlabel
    if ylabel is not None:
        yfeat = ylabel
    ax.set_xlabel(xfeat)
    ax.set_ylabel(yfeat)
    return ax


def plot_bldtype_by_proj(Xsc, y, CAfx, n_components, tt, evfx='', name='PC'):
    """
    Project to lower dimensions.
    """

    n_components = n_components
    sc, nplot = 5, n_components
    fig, axes = plt.subplots(5, nplot, figsize=(sc*2, sc*n_components))
    fig.tight_layout(pad=1.5)

    for ib, bldtype in enumerate(IBLDTYPES):
        ss = np.s_[:]
        ts = np.s_[y == ib]
        pca1 = CAfx(n_components=n_components)  # n_components)
        pca1.fit(Xsc[ss][ts])  # calculate PCA for Xsc
        # Xt = pca1.transform(Xsc[ss][ts])
        if evfx != '':
            ev = evfx(pca1)
            for i, ax in enumerate(axes[ib]):
                ax.imshow(pca1.components_[i].reshape(tt, tt), cmap='gray')
                ax.set_title('type: {} {}: {} exp_var: {}'.format(
                    name, bldtype, i, np.round(ev[i], 2)))
        else:
            for i, ax in enumerate(axes[ib]):
                ax.imshow(pca1.components_[i].reshape(tt, tt), cmap='gray')
                ax.set_title('type: {} {}: {}'.format(
                    bldtype, name, i))


def plot_bldtype_by_inv_proj(Xsc, y, CAfx, n_components, tt, name='PC'):
    """Reconstruct from lower dimension projection with selection of PCs.
    """
    n_components = n_components
    sc, nplot = 20, 1
    fig, axes = plt.subplots(nplot, 5, figsize=(sc*nplot, sc*nplot))
    for ib, bldtype in enumerate(IBLDTYPES):
        ss = np.s_[:]
        pca1 = CAfx(n_components=n_components)
        pca1.fit(Xsc[ss][y == ib])  # calculate PCA for Xsc
        Xt = pca1.transform(Xsc[ss][y == ib])
        components = Xt.shape[1]
        projected = pca1.inverse_transform(Xt)
        axes[ib].imshow(projected[0].reshape(tt, tt), cmap='gray')
        axes[ib].set_title('{} {}, {}s: {}'.format(
            bldtype, ib, name, components))


def plot_pairwise_feat(Xsc, y):
    """
    # 1=envelope, 2=max cooling, 3=equip eff
    """
    cols = ['blue', 'purple', 'red']
    marks = ['.', '1', 'x']
    sc, nplot = 2, 3
    fig, ax = plt.subplots(1, nplot, figsize=(sc*nplot, sc))
    fig.tight_layout(pad=2.0)
    for i in range(nplot):
        reduced_data = Xsc[:, FEAT_LABELS_IDX[i]]
        for j in range(3):
            idx = np.where(y == j)
            ax[i].scatter(
                reduced_data[idx, 0],
                reduced_data[idx, 1],
                color=cols[j], marker=marks[j])
            ax[i].grid()
        # ax[i].set_title()
        ax[i].set_xlabel(FEAT_LABELS[i][0])
        ax[i].set_ylabel(FEAT_LABELS[i][1])


def plot_pairwise_feat_by_proj(Xsc, y, CAfx, name='PC', fig=None, ax=None):
    """Plot features pairwise by projection.
    """
    sc, nplot = 2, 3
    if ax is None:
        fig, ax = plt.subplots(2, nplot, figsize=(sc*nplot, sc*2))
        fig.tight_layout(pad=2.0)
    n_comp = 2
    for i, (label, label_idx) in enumerate(zip(FEAT_LABELS, FEAT_LABELS_IDX)):
        ss = np.s_[:, label_idx]
        pca1 = CAfx(n_components=n_comp)
        pca1.fit(Xsc[ss])  # calculate PCA for Xsc
        Xt = pca1.transform(Xsc[ss])
        # pp(pca1.explained_variance_)
        ax[0][i].scatter(Xt[:, 0], Xt[:, 1], c=y, cmap='jet', marker='x')
        ax[0][i].grid()

        # Plot text
        xfeat, yfeat = label
        title_txt = name + 'A'
        title_txt = title_txt + ' ' + xfeat + ' vs ' + yfeat
        ax[0][i].set_title(title_txt)
        ax[0][i].set_xlabel(name + '1')
        ax[0][i].set_ylabel(name + '2')

        ax[1][i].scatter(Xt[:, 0], np.zeros(y.shape),
                         c=y, cmap='jet', marker='x')
        ax[1][i].set_ylim([-0.25, 0.25])


def calc_metrics(model, output):
    """Print clustring metrics.
    This function is referenced from:
    https://medium.com/@joel_34096/k-means-clustering-for-image-classification-a648f28bdc47
    """
    print('Number of clusters is {}'.format(model.n_clusters))
    print('Inertia: {}'.format(model.inertia_))
    print('Homogeneity: {}'.format(
        metrics.homogeneity_score(output, model.labels_)))


def shuffle_Xy(Xtrain, ytrain):
    """Shuffle data."""
    ridx = np.arange(Xtrain.shape[0])
    np.random.shuffle(ridx)
    XX = Xtrain[ridx]
    yy = ytrain[ridx]
    return XX, yy


def nn_performance_data(nn, Xtrain_data, ytrain_data, Xval_data, yval_data):
    start_time = time.time()
    nn.fit(Xtrain_data, ytrain_data)
    secs = time.time() - start_time
    train_gd_f1 = f1_score(ytrain_data, nn.predict(
        Xtrain_data), average='macro')
    val_gd_f1 = f1_score(yval_data, nn.predict(Xval_data), average='macro')
    return train_gd_f1, val_gd_f1, len(nn.fitness_curve),  nn.loss, secs


def learning_curve_df(nn, cv, train_sizes, XX, yy, df_path):
    """Build learning curve for model.
    # Code i modified from scikit learning_curve method here:
    # https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/model_selection/_validation.py#L1105
    """
    XX, yy = shuffle_Xy(XX, yy)
    cv = check_cv(cv, yy, classifier=True)
    cv_iter = list(cv.split(XX, yy, None))

    # Building learning curve batches
    n_max_training_samples = len(cv_iter[0][0])
    train_sizes_abs = _translate_train_sizes(
        train_sizes, n_max_training_samples)
    train_test_proportions = []
    for train, test in cv_iter:
        for n_train_samples in train_sizes_abs:
            train_test_proportions.append(
                (train[:n_train_samples], test))

    learn_crvs, learn_loss = [], []
    learn_size, learn_time = [], []
    train_f1, val_f1 = [], []
    for train_sample_idx, test_idx in train_test_proportions:
        # Get data
        Xtrain_data, ytrain_data = XX[train_sample_idx], yy[train_sample_idx]
        Xval_data, yval_data = XX[test_idx], yy[test_idx]

        # Run model on train
        _train_f1, _val_f1, crvs, loss, secs = \
            nn_performance_data(
                nn, Xtrain_data, ytrain_data, Xval_data, yval_data)

        learn_crvs.append(crvs)
        learn_loss.append(loss)
        # learn_pred_probs.append(pred_probs)
        learn_time.append(secs)
        learn_size.append(len(train_sample_idx))
        train_f1.append(_train_f1)
        val_f1.append(_val_f1)

    gd_df = pd.DataFrame(
        {'size': learn_size,
         'train_f1': train_f1,
         'cv_f1': val_f1,
         'calls': learn_crvs,
         'loss': learn_loss,
         'sec': learn_time}).sort_values('size')

    gd_learn_path = os.path.join('pkl', df_path)
    gd_df.to_pickle(gd_learn_path)
    return gd_learn_path


def learning_curve_plot(learn_df_path, model_name, train_sizes, figsize=(16, 4)):
    """Plot learning curve.

    # Modified from:
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    gd_df = pd.read_pickle(learn_df_path)
    gd_df_mean = gd_df.groupby('size').mean()
    gd_df_std = gd_df.groupby('size').std()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # if not isinstance(axes, (tuple, list, np.ndarray)):
    #    axes = [axes]

    # Plot learning curve (samples vs score for cv/training)
    train_scores_mean = gd_df_mean['train_f1']
    train_scores_std = gd_df_std['train_f1']
    test_scores_mean = gd_df_mean['cv_f1']
    test_scores_std = gd_df_std['cv_f1']
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("f1-weighted score")
    axes[0].set_title("Learning Curves ({})".format(model_name))

    # Plot scalability (samples vs times)
    fit_times_mean = gd_df_mean['sec']
    fit_times_std = gd_df_std['sec']

    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Wall Clock Time for Fit (seconds)")
    axes[1].set_title("Scalability of the model")

    # Plot scalability2 (samples vs calls)
    fit_times_mean = gd_df_mean['sec']
    fit_times_std = gd_df_std['sec']

    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Wall Clock Time for Fit (seconds)")
    axes[2].set_ylabel("f1-weighted score")
    axes[2].set_title("Performance of the model")

    for k in range(3):
        if k == 1:
            continue
        axes[k].set_yticks(np.arange(0.0, 1.2, 0.1))

    return axes
