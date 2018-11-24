import pandas as pd
import numpy as np
import matplotlib.pyplot as mplot
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

mplot.style.use("ggplot")

data = pd.read_excel("Static/basedehc.xlsx")

print("Base de données : ")
print(data)

################################
# Clustering
################################

# Méthode 1
################
# Nous allons de créer des clusters en prenant comme features, non pas chaque actifs (fond + spx) mais en prenant
# chaque dates, les observations étant les actifs.
# Notre dataset est composé de 15 actifs (13 fonds, SPX, Bond US), et 180 observations pour chaque actif

######################
# Création du Dataset
######################

dataset = data.T

print("Dataset :")
print(dataset.iloc[:-1, :])

results_1 = list()
for i in range(2, int(len(dataset) - 1)):
    result = dict()
    KNN = KMeans(n_clusters=i)
    KNN.fit(dataset.iloc[:-1, :])

    cluster_centers = KNN.cluster_centers_
    labels = KNN.labels_
    wss = KNN.inertia_
    euclidian_silhouette = \
        silhouette_score(dataset.iloc[:-1, :], labels, metric='euclidean')
    manhattan_silouhette = \
        silhouette_score(dataset.iloc[:-1, :], labels, metric='manhattan')
    minkowski_silouhette = \
        silhouette_score(dataset.iloc[:-1, :], labels, metric='minkowski')
    euclidian_samples_silhouette = \
        silhouette_samples(dataset.iloc[:-1, :], labels, metric='euclidean')
    manhattan_samples_silouhette = \
        silhouette_samples(dataset.iloc[:-1, :], labels, metric='manhattan')
    minkowski_samples_silouhette = \
        silhouette_samples(dataset.iloc[:-1, :], labels, metric='minkowski')
    cluster_frequency = pd.Series(labels).value_counts()

    result.update(centroides=cluster_centers)
    result.update(labels=labels)
    result.update(wss=wss)
    result.update(n_cluster=i)
    result.update(euclidian=euclidian_silhouette)
    result.update(manhattan=manhattan_silouhette)
    result.update(minkowski=minkowski_silouhette)
    result.update(euclidian_samples=euclidian_samples_silhouette)
    result.update(manhattan_samples=manhattan_samples_silouhette)
    result.update(minkowski_samples=minkowski_samples_silouhette)
    result.update(cluster_frequency=cluster_frequency)
    results_1.append(result)

arbitrary_score = results_1[5]['minkowski']
cluster_samples_scores_1 = [x["minkowski_samples"] for x in results_1]
cluster_frequencies_1 = [x["cluster_frequency"] for x in results_1]
selected_model_params_1 = \
    [(x["n_cluster"], x["labels"], x["centroides"]) for x in results_1 if x['minkowski'] == arbitrary_score]
(n_clusters_opt_1, labels_opt_1, label_centers_opt_1) = selected_model_params_1[0]
clusterised_dataset_1 = dataset.join(pd.DataFrame(labels_opt_1, columns=["cluster"], index=dataset.index[:-1]))

print("Résultats du modèle à " + str(n_clusters_opt_1) + " clusters selectionne arbitrairement : ")
print("Score d'erreur avec la distance Euclidienne : " + str([x["euclidian"] for x in results_1 if x['minkowski'] == arbitrary_score]))
print("Score d'erreur avec la distance de Minkowski : " + str([x["minkowski"] for x in results_1 if x['minkowski'] == arbitrary_score]))
print("Score d'erreur avec la distance de Manhattan : " + str([x["manhattan"] for x in results_1 if x['minkowski'] == arbitrary_score]))
print("DataSet Optimal Classifié : ")
print(clusterised_dataset_1)

# Résultat :
# Nombre clusters : 7
# le SPX et les BONDS sont dans leurs propres clusters
#   SPX : Cluster N° 4
#   Gov_US : Cluster N° 3

# Les fonds, eux, appartiennent aux groupe 0, 1, 2, 5, 6 :
#   Funds of Funds, Relative Value, Long / Short Equity, Event Driven, Distressed Securities, Convertible Arbitrage : Cluster N° 0
#   CTA Global : Cluster N° 1
#   Short Selling : Cluster N° 2
#   Emerging Markets : Cluster N° 5
#   Merger Arbitrage, Global Macro, Fixed Income Arbitrage, Equity Market Neutral : Cluster N° 6

# ==> Le SPX et les Bonds sont bien dans leurs propres clusters, c'est le résultat qu'on esperais

# Affichons le return moyen de chaque clusters :

for index, cluster in enumerate(label_centers_opt_1):
    fig = mplot.figure(figsize=(10, 5))
    axes = mplot.axes()
    members = clusterised_dataset_1[clusterised_dataset_1["cluster"] == index].index
    mplot.title("Cluster N° " + str(index) + " members : " + ' / '.join(members))
    axes.plot(cluster, 'r-', label="Cluster N°" + str(index))
    axes.plot(np.full((len(cluster), 1), np.mean(cluster)), 'b--', label="Mean returns")
    axes.plot(np.full((len(cluster), 1), np.mean(cluster) + np.std(cluster)), 'g-.', label="Bollinger Bands (x1 std)")
    axes.plot(np.full((len(cluster), 1), np.mean(cluster) - np.std(cluster)), 'g-.')
    mplot.legend(loc='lower right')
    mplot.xlabel('t')
    mplot.ylabel('Return')
    mplot.show()

# Méthode 2
################
# On vas chercher à créer un indicateur, pour chaque classe d'actifs, de sa dynamique sur la période considérée.
# Nous allons considérer 5 périodes, en les découpant en fonction de l'indicateur de récession

############################
# Création du Dataset
############################

recession_index = dataset.loc["rec_us", 0]
final_dataset = pd.DataFrame()
periods_days = list()
one_period_data = pd.DataFrame()
i_periods = 1

# for row in dataset.iterrows():
for period in dataset.columns:
    if dataset.loc["rec_us", period] == recession_index:
        one_period_data[period] = dataset[period]

    if dataset.loc["rec_us", period] != recession_index or \
            dataset.iloc[:, period].equals(dataset.iloc[:, 179]):
        final_dataset["Periode" + str(i_periods)] = one_period_data.mean(axis=1)
        periods_days.append(len(one_period_data.columns))
        one_period_data = pd.DataFrame()
        one_period_data[period] = dataset[period]
        recession_index = dataset.loc["rec_us", period]
        i_periods += 1

print(final_dataset)

############################
# Clustering
############################

results_2 = list()
for i in range(2, int(len(final_dataset) - 1)):
    result = dict()
    KNN = KMeans(n_clusters=i)
    KNN.fit(final_dataset.iloc[:-1, :])

    cluster_centers = KNN.cluster_centers_
    labels = KNN.labels_
    wss = KNN.inertia_
    euclidian_silhouette = \
        silhouette_score(final_dataset.iloc[:-1, :], labels, metric='euclidean')
    manhattan_silouhette = \
        silhouette_score(final_dataset.iloc[:-1, :], labels, metric='manhattan')
    minkowski_silouhette = \
        silhouette_score(final_dataset.iloc[:-1, :], labels, metric='minkowski')
    euclidian_samples_silhouette = \
        silhouette_samples(dataset.iloc[:-1, :], labels, metric='euclidean')
    manhattan_samples_silouhette = \
        silhouette_samples(dataset.iloc[:-1, :], labels, metric='manhattan')
    minkowski_samples_silouhette = \
        silhouette_samples(dataset.iloc[:-1, :], labels, metric='minkowski')

    cluster_frequency = pd.Series(labels).value_counts()

    result.update(centroides=cluster_centers)
    result.update(labels=labels)
    result.update(wss=wss)
    result.update(n_cluster=i)
    result.update(euclidian=euclidian_silhouette)
    result.update(manhattan=manhattan_silouhette)
    result.update(minkowski=minkowski_silouhette)
    result.update(euclidian_samples=euclidian_samples_silhouette)
    result.update(manhattan_samples=manhattan_samples_silouhette)
    result.update(minkowski_samples=minkowski_samples_silouhette)
    result.update(cluster_frequency=cluster_frequency)
    results_2.append(result)

scores = [x["minkowski"] for x in results_2]
cluster_samples_scores_2 = [x["minkowski_samples"] for x in results_2]
cluster_frequencies_2 = [x["cluster_frequency"] for x in results_2]
arbitrary_score = results_2[5]['minkowski']
selected_model_params_2 = \
    [(x["n_cluster"], x["labels"], x["centroides"]) for x in results_2 if x['minkowski'] == arbitrary_score]
(n_clusters_opt_2, labels_opt_2, label_centers_opt_2) = selected_model_params_2[0]
clusterised_dataset_2 = final_dataset.join(pd.DataFrame(labels_opt_2, columns=["cluster"], index=dataset.index[:-1]))

print("Résultats du modèle à " + str(n_clusters_opt_2) + " clusters selectionne arbitrairement : ")
print("Score d'erreur avec la distance Euclidienne : " + str([x["euclidian"] for x in results_2 if x['minkowski'] == arbitrary_score]))
print("Score d'erreur avec la distance de Minkowski : " + str([x["minkowski"] for x in results_2 if x['minkowski'] == arbitrary_score]))
print("Score d'erreur avec la distance de Manhattan : " + str([x["manhattan"] for x in results_2 if x['minkowski'] == arbitrary_score]))
print("DataSet Optimal Classifié : ")
print(clusterised_dataset_2)

################################################
# Optimisation du nombre de clusters
################################################
# Nous allons chercher à determiner le nombre de clusters
# optimaux, CAD qui vas minimiser la somme des
# carrés des distances intra-clusters de chaque
# observation du dataset.
# Nous utiliserons 2 méthode (Elbow et méthode de la Sihlouette)
# Et 3 distances (Euclidiean, Manhattan et Minkowski)

# Méthode d'Elbow
################################################
# Consistes à afficher sur un graphique le nombre de clusters par
# ordre croissant sur l'axe X , correspondant à un score
# (somme total de la distance intra clusters
# au carré de chaque observations), sur l'axe des Y.
# On choisit le nombre de clusters à l'endroit où un "coude"
# apparait sur le graphe : L'augmentation du nombre de clusters ne fais
# plus diminuer de beaucoup le score

# Dataset 1

# Euclidian Distance

scores1 = [x["euclidian"] for x in results_1]

fig = mplot.figure(figsize=(10, 5))
axes = mplot.axes()

mplot.plot(range(2, len(scores1) + 2), scores1, 'o-.r', label="Sum of Euclidian intra-cluster distance")
mplot.title("score du clustering en fonction du nombre de clusters")
mplot.xlabel("Nombres de Clusters")
mplot.ylabel("Score d'erreur")
mplot.legend(loc='upper right')

mplot.show()

# Manhattan Distance

scores1 = [x["manhattan"] for x in results_1]

fig = mplot.figure(figsize=(10, 5))
axes = mplot.axes()

mplot.plot(range(2, len(scores1) + 2), scores1, 'o-.r', label="Sum of Manhattan intra-cluster distance")
mplot.title("score du clustering en fonction du nombre de clusters")
mplot.xlabel("Nombres de Clusters")
mplot.ylabel("Score d'erreur")
mplot.legend(loc='upper right')

mplot.show()

# Minkowski Distance

scores1 = [x["minkowski"] for x in results_1]

fig = mplot.figure(figsize=(10, 5))
axes = mplot.axes()

mplot.plot(range(2, len(scores1) + 2), scores1, 'o-.r', label="Sum of Minkowski intra-cluster distance")
mplot.title("score du clustering en fonction du nombre de clusters")
mplot.xlabel("Nombres de Clusters")
mplot.ylabel("Score d'erreur")
mplot.legend(loc='upper right')

mplot.show()

# On peut voir que jusqu'à 4 clusters, le score baisse très rapidement : on ne prendras pas moins de 4 clusters
# De la même manière, à partir de 8 clusters, celà n'a presque plus aucun effet sur le score du modèle d'ajouter
# un cluster supplémentaire : on ne prendras pas plus de 8 clusters
# On décide de prendre le "coude" du graphique :
# La méthode d'Elbow donne un modèle optimal à 7 clusters pour le dataset 1

# Dataset 2

# Euclidian Distance

scores2 = [x["euclidian"] for x in results_2]

fig = mplot.figure(figsize=(10, 5))
axes = mplot.axes()

mplot.plot(range(2, len(scores2) + 2), scores2, 'o-.r', label="Sum of Euclidian intra-cluster distance")
mplot.title("score du clustering en fonction du nombre de clusters")
mplot.xlabel("Nombres de Clusters")
mplot.ylabel("Score d'erreur")
mplot.legend(loc='upper right')

mplot.show()

# Manhattan Distance

scores2 = [x["manhattan"] for x in results_2]

fig = mplot.figure(figsize=(10, 5))
axes = mplot.axes()

mplot.plot(range(2, len(scores2) + 2), scores2, 'o-.r', label="Sum of Manhattan intra-cluster distance")
mplot.title("score du clustering en fonction du nombre de clusters")
mplot.xlabel("Nombres de Clusters")
mplot.ylabel("Score d'erreur")
mplot.legend(loc='upper right')

mplot.show()

# Minkowski Distance

scores2 = [x["minkowski"] for x in results_2]

fig = mplot.figure(figsize=(10, 5))
axes = mplot.axes()

mplot.plot(range(2, len(scores2) + 2), scores2, 'o-.r', label="Sum of Minkowski intra-cluster distance")
mplot.title("score du clustering en fonction du nombre de clusters")
mplot.xlabel("Nombres de Clusters")
mplot.ylabel("Score d'erreur")
mplot.legend(loc='upper right')

mplot.show()

# On peut voir que jusqu'à 6 clusters, le score baisse très rapidement :
# on ne prendras pas moins de 6 clusters
# On observe un autre point d'inflextion pour 11 clusters :
# on ne prendras pas plus de 11 clusters
# On décide de prendre le deuxième point d'inflextion : 11 clusters

# Méthode de la Silhouette
#################################################
# Nous allons :
#      * Calculer un score de silhouette pour chaque clusters
#      * Caluler un score de silhouette globbal du modèle declustering
# On cherche ensuite a avoir le score de silhouette global le plus
# élevé possible, tout en ayant des score de
# silhouette individuel assez homogènes, et surtout, non-négatifs.

# Dataset 1

for nb_clusters in range(0, len(scores1)):
    fig = mplot.figure(figsize=(10, 5))
    axes = mplot.axes()
    n_features = 15
    mplot.ylim(-1, 1)
    mplot.ylabel('Silhouette Score')
    mplot.xlabel('Cluster numbers')

    for n_cluster, freq in enumerate(cluster_frequencies_1[nb_clusters]):
        mplot.bar(n_cluster, cluster_samples_scores_1[nb_clusters][n_cluster], width=freq / n_features)

    (xlim_min, xlim_max) = axes.get_xlim()
    mplot.plot(np.linspace(xlim_min, xlim_max, nb_clusters + 2),
               [scores1[nb_clusters] for i in range(0, len(cluster_frequencies_1[nb_clusters]))],
               '-.g', label="average silhouette")
    mplot.legend(loc='upper right')
    mplot.title("Cluster Number : " + str(nb_clusters + 2))
    mplot.show()

# Dataset 2

for nb_clusters in range(0, len(scores2)):
    fig = mplot.figure(figsize=(10, 5))
    axes = mplot.axes()
    n_features = 15
    mplot.ylim(-1, 1)
    mplot.ylabel('Silhouette Score')
    mplot.xlabel('Cluster numbers')

    for n_cluster, freq in enumerate(cluster_frequencies_2[nb_clusters]):
        mplot.bar(n_cluster, cluster_samples_scores_2[nb_clusters][n_cluster], width=freq / n_features)

    (xlim_min, xlim_max) = axes.get_xlim()
    mplot.plot(np.linspace(xlim_min, xlim_max, nb_clusters + 2),
               [scores2[nb_clusters] for i in range(0, len(cluster_frequencies_2[nb_clusters]))],
               '-.g', label="average silhouette")
    mplot.legend(loc='upper right')
    mplot.title("Cluster Number : " + str(nb_clusters + 2))
    mplot.show()

# On cherche donc un nombre de clusters optimal tel que :
#   - Les barres aient toutes à peu près la même taille
#   - La silhouette moyenne soit la plus haute possible
#   - Il n'y ai aucune sihouette négatives (indique que certaines observations du clusters sont dans le mauvais cluster)

# Pour des modèles à 2 ou 3 clusters, ces derniers ne sont clairement pas homogènes en effectifs On cherche donc un
# nombre de cluster optimal strictement supérieurs à 3.
# Pour des modèles à plus de 4 clusters, Il y a plusieures silhouettes négatives On cherche donc un nombre de cluster
# optimal strictement supérieurs à 3.
# Le modèle à 4 clusters a un silhouette moyenne (0.3) a peu près équivalente à la silhouette maximum (donné par le
#  modèle à 3 clusters ~= 0.4)
# Notre modèle optimal est donc un modèle à 4 clusters

##################################################
# Conclusion sur le nombre de cluster optimals
##################################################

#   - La méthode d'Elbow donne un nombre de clusters optimal de 6 clusters
#   - La méthode de la Silouhette donne un nombre de clusters optimal de 4 clusters
# -->     La méthode de la Silhouette étant plus rigoureuse,
# on sélectionne le nombre de clusters optimal
# donné par cette méthode

#############################################################
# Visualisation du Dataset clusterisé suivant le modèle optimal
#############################################################

# Dataset 1

# Clustering
#################

n_cluster_opt = 6

KNN = KMeans(n_clusters=n_cluster_opt)
KNN.fit(dataset.iloc[:-1,:])

cluster_centers = KNN.cluster_centers_
labels = KNN.labels_
wss = KNN.inertia_

clusterised_dataset = dataset.join(pd.DataFrame(labels, columns=["cluster"], index=dataset.index[:-1]))
clusterised_dataset

# Cluster Returns Visualisation
#################################

fig = mplot.figure(figsize=(20, 10))
axes = mplot.axes()

mplot.title("Return moyen des 5 périodes, par cluster")
mplot.xlabel("Periode N°")
mplot.ylabel("Return")
for index, cluster in enumerate(cluster_centers):
    cluster_members = clusterised_dataset[clusterised_dataset["cluster"] == int(index)].index
    mplot.plot(range(1, len(dataset.columns) + 1), cluster, '-',
               label="cluster N° " + str(index + 1) + " Members : " + " / ".join(cluster_members))

mplot.legend(loc="lower left", prop={'size': 10})
mplot.show()

# Dataset 2

# Clustering
#################

n_cluster_opt = 4

KNN = KMeans(n_clusters=n_cluster_opt)
KNN.fit(final_dataset.iloc[:-1, :])

cluster_centers = KNN.cluster_centers_
labels = KNN.labels_
wss = KNN.inertia_

clusterised_dataset = final_dataset.join(pd.DataFrame(labels, columns=["cluster"], index=dataset.index[:-1]))
print(clusterised_dataset)

# Cluster Returns Visualisation
#################################

fig = mplot.figure(figsize=(20, 10))
axes = mplot.axes()

mplot.title("Return moyen des 5 périodes, par cluster")
mplot.xlabel("Periode N°")
mplot.ylabel("Return")
for index, cluster in enumerate(cluster_centers):
    cluster_members = clusterised_dataset[clusterised_dataset["cluster"] == int(index)].index
    mplot.plot(range(1, len(final_dataset.columns) + 1), cluster, 'o-',
               label="cluster N° " + str(index + 1) + " Members : " + " / ".join(cluster_members))

mplot.legend(loc="lower left", prop={'size': 10})
mplot.show()

# Résultats
#############
# On peut voir que le SPX appartient à son propre cluster :
# SPX : Cluster N° 4
# Les bonds, quant à eux, partagent un clusters avec les fonds "CTA Global" :
# US Bonds + CTA Global : Cluster N° 5
# Les fonds (hors CTA Global), eux, appartiennent aux groupe 0, 1, 2 ou 3 :
# Distressed Securites / Emerging Markets : Cluster N° 0
# Short Selling : Cluster N° 1
# Global Macro / Merger Arbitrage : Cluster N° 2
# Convertible Arbitrage / Equity Market Neutral / Event Driven / Fixed Income Arbitrage / Long Short Equity / Relative Value / Funds of Funds : Cluster N° 3

# Conclusion
################
# Les Hedge Funds permettent ils de divérsifier un portefeuille ?
# Lorsqu'on parle de diversification d'un portefeuille, on parle de composer un portefeuille de titres financiers,
# où ceux-ci n'appartiendraient pas tous à la même catégorie, ou classes d'actifs, classes qui sont elles même définies
# par une dynamique temporelle et un profil risque/rentabilité qui leur sont propres.
# On sait que, lorsqu'un portefeuille est constitué d'un seul type d'actif financier, c'est à dire composé d'une seule
# ou de peu de classe d'actifs différentes, alors on peut diminuer le niveau de risque du portefeuille, qui, par
# définition d'une classe d'actifs, présente des rendement plutôt corréllés, en composant son portefeuille de manière
# plus divérsifié par rapport aux classes d'actifs qui le compose, tout en conservant le même niveau de rentabilité.
# En effet, un portefeuille faiblement diversifé présente un risque dans la mesure où, si la classe d'actifs dans
# laquelle nous avons investi baisse de prix pour des raisons extra-ordinaires, on ne peut pas compenser cette perte
# par une position sur une autre classe d'actifs. On peut grossièrement traduire celà par le risque de "Mettre tout
# ses oeufs dans le même panier"
#
# Ainsi, pour affirmer que les Hedge Funds permettent de diversifier un portefeuille, il faudrait prouver que ces
# derniers sont effectivement une classe d'actifs propres, ayant une dynamique temorelle et un profile
# rentabilité/risque qui leurs sont propres. On prouverait ainsi que ceux ci représentent une (ou plusieurs) classes
# d'actifs à eux seuls, offrant donc à l'investisseur une autre possibilité d'allocation dans
# l'espace rentabilité/risque et donc in fine, lui permettent de diversifier son portefeuille.
#
# A cet fin, nous avons donc voulu mettre en évidence des clusters sur les rendement de différentes classes actifs,
# notamment des fonds, et sur une période de temps de 180 jours ouvrés, en cherchant à démontrer que les fonds se
# distinguent bien des autres classes d'actifs que sont les actions (SPX), et les bonds gouvernementaux (US Bonds).
#
# Au terme de notre, analyse, en concluant que le nombre de clusters optimaux sur le Dataset dont nous disposions,
# était de 6 clusters, et en observant qu'effectivement, le SPX d'une part, et les US Bonds d'autre part, constituent
# deux clusters différent de ceux dans lesquels sont contenus les fonds (4 autres clusters), nous démontrons par là
# même que les fonds sont bien une classe d'actifs à part, présentant un profil risque/rentabilité différents des
# autres classes d'actifs "classiques" (Stock et Bonds).
#
# Ainsi, les Hedge Funds aident effectivement à la diversification
# d'un portefeuille.
