import pandas as pd
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from adjustText import adjust_text
from champ_role.champ_role import champ_role


# constant
random_state = 42

# load data
data = pd.read_feather("../mastery_data_1m.feather")
champion_names = data.columns[1:]
data_numeric = data.drop(columns=["PUUID"])

# TESTING - only take first 50k rows
# data_numeric = data_numeric.head(1000000)

# color mapping based on roles
role_colors = {
    'Top': 'blue',
    'Jungler': 'red',
    'Mid': 'green',
    'AD Carry': 'purple',
    'Support': 'orange',
}

# normalising data
scaler = StandardScaler()
data_normalised = scaler.fit_transform(data_numeric)


def visualise(embeddings_2d, method_name, t):
    plt.figure(figsize=(12, 10))
    texts = []

    for i, champion in enumerate(champion_names):
        role = champ_role.get(champion, "Unknown")
        color = role_colors.get(role, "gray")
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color, alpha=0.5)
        text = plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], champion, fontsize=8, ha='right', va='top',
                        color=color)
        texts.append(text)

    # apply adjust_text to improve text placement to prevent overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=0.5),
                force_text=(0.5, 0.5))

    # legend for colors
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=role)
                      for role, color in role_colors.items()]
    plt.legend(handles=legend_handles, loc="best")

    # title and axes labels
    title_text = f"{method_name} visualisation of champ embeddings by role\n"
    if method_name in ["t-SNE"]:
        title_text += f"using {method_name} random_state={random_state}\n"
    title_text += f"Processing duration: {time.time() - t:.2f} seconds"
    plt.title(title_text)
    plt.xlabel(f"{method_name} comp 1")
    plt.ylabel(f"{method_name} comp 2")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def visualise_3d(embeddings, method_name):
    fig = go.Figure()
    for i, champion in enumerate(champion_names):
        role = champ_role.get(champion, "Unknown")
        color = role_colors.get(role, "gray")

        fig.add_trace(go.Scatter3d(
            x=[embeddings[i, 0]],
            y=[embeddings[i, 1]],
            z=[embeddings[i, 2]],
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                opacity=0.8
            ),
            text=champion,
            name=champion,
            hoverinfo='text'
        ))
    fig.update_layout(
        title=f'{method_name} Visualisation',
        scene=dict(
            xaxis=dict(title=''),
            yaxis=dict(title=''),
            zaxis=dict(title=''),
        )
    )
    fig.show()


def tsne():
    t = time.time()
    tsne = TSNE(n_components=2, random_state=random_state)
    embeddings_2d = tsne.fit_transform(data_normalised)
    visualise(embeddings_2d, "t-SNE", t)


def pca():
    t = time.time()
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(data_normalised)
    visualise(embeddings_2d, "PCA", t)


def pca_3d():
    pca = PCA(n_components=3)  # 3d
    embeddings_3d = pca.fit_transform(data_normalised)
    return embeddings_3d


def autoencoder():
    t = time.time()
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(data_normalised.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='relu'))  # encoder layer
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(data_normalised.shape[1], activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(data_normalised, data_normalised, epochs=500, batch_size=4096, verbose=1)
    encoder = Sequential(model.layers[:3])
    embeddings_2d = encoder.predict(data_normalised)
    visualise(embeddings_2d, "Autoencoders", t)


if __name__ == "__main__":
    # tsne()
    # pca()
    # autoencoder()
    visualise_3d(pca_3d(), "PCA 3D")
