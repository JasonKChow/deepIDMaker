import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import audioIO
import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def get_reps(model, files, stimDir):
    reps = np.empty((len(files), 1024))
    for i, file in enumerate(files):
        clip = audioIO.load_wav(os.path.join(stimDir, file))
        clip = tf.expand_dims(clip, axis=0)
        embeddings = model(clip)["embedding"]

        reps[i, :] = np.mean(embeddings, axis=0)
    return reps


def get_identities(files):
    identities = [name[2:-4] for name in files]

    # Split on the letter
    identities = [re.split(r"([a-zA-Z])", name)[0] for name in identities]

    # Add FS back to everyone
    identities = ["FS" + name for name in identities]

    return identities


def create_sim_matrix(df):
    # Create similarity matrix
    simMatrix = np.empty((len(df), len(df)))

    for i in range(len(df)):
        for j in range(len(df)):
            simMatrix[i, j] = np.sum((df["rep"][i] - df["rep"][j]) ** 2)

    return simMatrix


def plot_sim_matrix(simMatrix, outpath):
    plt.imshow(simMatrix)
    plt.savefig(outpath)


if __name__ == "__main__":
    np.random.seed(2022)

    stimDir = "./stimuli/"
    files = audioIO.get_files(stimDir)
    identities = get_identities(files)

    model = hub.load("https://tfhub.dev/google/trillsson5/1")

    reps = get_reps(model, files, stimDir)

    df = pd.DataFrame(
        data={
            "file": files,
            "identity": identities,
            "rep": [rep for rep in reps],
        }
    )
    # Sort df by identity
    df = df.sort_values(by=["identity"])

    simMatrix = create_sim_matrix(df)
    plot_sim_matrix(simMatrix, "./laughSimMatrix.png")

    # Get similarity values from lower triangle
    simValues = simMatrix[np.tril_indices(len(simMatrix), -1)]
    simLow = np.mean(simValues) - np.std(simValues)
    simHigh = np.mean(simValues) + np.std(simValues)
    simStd = np.std(simValues)

    # Loop through identity, make one easy, medium, hard trial
    combosDf = pd.DataFrame(
        columns=[
            "identity",
            "targetFile",
            "correctFile",
            "foilFile",
            "diffScore",
            "diffBracket",
        ]
    )
    for identity in df["identity"].unique():
        # Get all files for this identity
        identityFiles = df[df["identity"] == identity]["file"].values

        # Check if there are at least 2 files for this identity
        if len(identityFiles) < 2:
            continue

        # Pick a random file for this identity
        targetFile = np.random.choice(identityFiles)

        # Pick a file that is not the target file
        nonTargetFile = np.random.choice(
            identityFiles[identityFiles != targetFile]
        )

        # Calculate similarity between target and non-target
        targetRep = df[df["file"] == targetFile]["rep"].values[0]
        nonTargetRep = df[df["file"] == nonTargetFile]["rep"].values[0]
        corrSim = np.sum((targetRep - nonTargetRep) ** 2)

        # Calculate similarity between target and all other identity files
        tmpDf = df[df["identity"] != identity]
        for i, row in tmpDf.iterrows():
            tmpDf.loc[i, "sim"] = np.sum((targetRep - row["rep"]) ** 2)

        # Sort
        tmpDf = tmpDf.sort_values(by=["sim"])

        # Pick the most similar file
        hardFile = tmpDf.iloc[0]["file"]
        hardSim = tmpDf.iloc[0]["sim"]
        combosDf = combosDf.append(
            {
                "identity": identity,
                "targetFile": targetFile,
                "correctFile": nonTargetFile,
                "foilFile": hardFile,
                "diffScore": corrSim - hardSim,
                "diffBracket": "hard",
            },
            ignore_index=True,
        )

        # Pick the most similar file
        easyFile = tmpDf.iloc[-1]["file"]
        easySim = tmpDf.iloc[-1]["sim"]
        combosDf = combosDf.append(
            {
                "identity": identity,
                "targetFile": targetFile,
                "correctFile": nonTargetFile,
                "foilFile": easyFile,
                "diffScore": corrSim - easySim,
                "diffBracket": "easy",
            },
            ignore_index=True,
        )

        # Pick the file that is closest to the similarity of the target and non-target
        tmpDf["diff"] = np.abs(tmpDf["sim"] - corrSim)
        mediumFile = tmpDf.sort_values(by=["diff"]).iloc[0]["file"]
        mediumSim = tmpDf.sort_values(by=["diff"]).iloc[0]["sim"]
        combosDf = combosDf.append(
            {
                "identity": identity,
                "targetFile": targetFile,
                "correctFile": nonTargetFile,
                "foilFile": mediumFile,
                "diffScore": corrSim - mediumSim,
                "diffBracket": "medium",
            },
            ignore_index=True,
        )

    # Sort trials based on difficulty
    combosDf = combosDf.sort_values(by=["diffScore"], ascending=False)
    trialsDf = pd.DataFrame(
        columns=[
            "TrialN",
            "Target1",
            "Choice1",
            "Choice2",
            "CorrRes",
            "DiffScore",
        ]
    )
    # Create actual trials with some repeat limits
    repLimit = 3
    for row in combosDf.iterrows():
        # Get information on how many times each clip was used anywhere
        clipCounts = trialsDf["Target1"].value_counts()
        clipCounts = clipCounts.append(
            trialsDf["Choice1"].value_counts()
        ).append(trialsDf["Choice2"].value_counts())
        clipCounts = clipCounts.groupby(clipCounts.index).sum()

        # Get a list of all clips used
        clipsUsed = trialsDf["Target1"].unique()
        clipsUsed = np.append(clipsUsed, trialsDf["Choice1"].unique())
        clipsUsed = np.append(clipsUsed, trialsDf["Choice2"].unique())

        # Check if any of the files have been used too many times
        if (
            (
                row[1]["targetFile"] in clipsUsed
                and clipCounts[row[1]["targetFile"]] >= repLimit
            )
            or (
                row[1]["correctFile"] in clipsUsed
                and clipCounts[row[1]["correctFile"]] >= repLimit
            )
            or (
                row[1]["foilFile"] in clipsUsed
                and clipCounts[row[1]["foilFile"]] >= repLimit
            )
        ):
            continue

        # Pick correct response at random
        corrRes = np.random.choice([1, 2])

        # Add this row to the trials df
        trialsDf = trialsDf.append(
            {
                "Target": row[1]["targetFile"],
                "Choice1": row[1]["correctFile"]
                if corrRes == 1
                else row[1]["foilFile"],
                "Choice2": row[1]["foilFile"]
                if corrRes == 1
                else row[1]["correctFile"],
                "CorrRes": corrRes,
                "DiffScore": row[1]["diffScore"],
            },
            ignore_index=True,
        )

    # Sort trials back to easy to hardest
    trialsDf = trialsDf.sort_values(by=["DiffScore"], ascending=True)

    # Add trialNumber column
    trialsDf["TrialN"] = np.arange(1, len(trialsDf) + 1)

    # Save as json
    trialsDf.to_json("trials.json", orient="records")
    trialsDf
