import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_feather('../mastery_data_1m.feather')


def info():
    print(df.info())
    print(df.describe())


def most_played_champs(df):
    """plot the top 5 most played champions and the top 5 least played champions"""
    df = df.drop(columns=['PUUID'])
    df = df.sum().sort_values(ascending=False)
    top_champs = df.head(5)
    bottom_champs = df.tail(5)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    top_champs.plot(kind='bar', ax=ax[0], color='b')
    ax[0].set_title('Top 5 Most Played Champions')
    ax[0].set_xlabel('Champions')
    ax[0].set_ylabel('Play Count')
    ax[0].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
    bottom_champs.plot(kind='bar', ax=ax[1], color='r')
    ax[1].set_title('Top 5 Least Played Champions')
    ax[1].set_xlabel('Champions')
    ax[1].set_ylabel('Play Count')
    ax[1].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()


def one_trick_champs(df):
    """a player is considered a one-trick if they have more than 70 % of their total mastery points on a single champion"""
    df = df.drop(columns=['PUUID'])
    total_mastery = df.sum(axis=1)
    max_mastery = df.max(axis=1)
    one_trick_ratio = max_mastery / total_mastery
    one_trick_players = df[one_trick_ratio >= 0.7]
    one_trick_sum = one_trick_players.sum()
    one_trick_sum_sorted = one_trick_sum.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    if not one_trick_sum_sorted.empty:  # in case criteria is too strict
        one_trick_sum_sorted.head(5).plot(kind='bar', color='purple')
        plt.title('Top 5 Most One-Tricked Champions')
        plt.xlabel('Champion')
        plt.ylabel('Total Mastery Points from One-Tricks')
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("no data available for plotting")


def plot_most_played_by_puuid_letter(df):
    """if your puuid starts with "a" you are like to play champion X the most"""
    filtered_df = df[df['PUUID'].str[0].str.lower().isin(list('abcdefghij'))]
    grouped = filtered_df.groupby(filtered_df['PUUID'].str[0].str.lower())
    most_played_by_letter = {}
    for letter, group in grouped:
        summed_mastery = group.drop(columns=['PUUID']).sum()
        most_played_champion = summed_mastery.idxmax()
        mastery_points = summed_mastery.max()
        most_played_by_letter[letter] = (most_played_champion, mastery_points)

    plt.figure(figsize=(12, 6))
    bar_locations = range(len(most_played_by_letter))
    bar_heights = [info[1] for info in most_played_by_letter.values()]
    bar_labels = [info[0] for info in most_played_by_letter.values()]
    letters = list(most_played_by_letter.keys())

    bars = plt.bar(bar_locations, bar_heights, tick_label=bar_labels, color='skyblue')

    for bar, letter in zip(bars, letters):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, letter, va='bottom')  # adjust positioning as needed

    plt.xlabel('Champion')
    plt.ylabel('Mastery Points')
    plt.title('Most Played Champion by First Letter of PUUID (A-J)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    # info()
    most_played_champs(df)
    one_trick_champs(df)
    plot_most_played_by_puuid_letter(df)