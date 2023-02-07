import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import audioIO
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def get_reps(model, files, stimDir):
    reps = np.empty((len(files), 1024))
    for i, file in enumerate(files):
        clip = audioIO.load_wav(os.path.join(stimDir, file))
        clip = tf.expand_dims(clip, axis=0)
        embeddings = model(clip)["embedding"]

        reps[i, :] = np.mean(embeddings, axis=0)
    return reps


def get_FS_identities(files):
    # Strip FS
    identities = [name[2:-4] for name in files]

    # Split on the letter
    identities = [re.split(r"([a-zA-Z])", name)[0] for name in identities]

    # Add FS back to everyone
    identities = ["FS" + name for name in identities]

    return identities


def create_sim_matrix(df, group):
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

    model = hub.load("https://tfhub.dev/google/trillsson5/1")
    # Get files
    stimDir = "./stimuli/"
    files = audioIO.get_files(stimDir)

    # Check if fs similarity matrix exists
    if os.path.exists("./fsSimMatrix.npy"):
        fsSimMatrix = np.load("./fsSimMatrix.npy")
        fsIdentities = pickle.load(open("./fsIdentities.pkl", "rb"))
    elif os.path.exists("./fsReps.npy"):
        fsReps = np.load("./fsReps.npy")
        fsIdentities = pickle.load(open("./fsIdentities.pkl", "rb"))
        fsSimMatrix = cosine_similarity(fsReps)
        del fsReps
        np.save("./fsSimMatrix.npy", fsSimMatrix)
    else:
        fsFiles = [file for file in files if file[:2] == "FS"]

        # Get identities
        fsIdentities = [name[2:-4] for name in fsFiles]
        fsIdentities = [
            re.split(r"([a-zA-Z])", name)[0] for name in fsIdentities
        ]
        fsIdentities = ["FS" + name for name in fsIdentities]

        # Combine files and identities
        fsIdentities = pd.DataFrame(
            {"file": fsFiles, "identity": fsIdentities}
        )

        # Get FS reps
        fsReps = get_reps(model, fsFiles, stimDir)

        # Get similarity
        fsSimMatrix = cosine_similarity(fsReps)

        # Save reps
        np.save("./fsReps.npy", fsReps)
        del fsReps

        # Save identity list
        pickle.dump(fsIdentities, open("./fsIdentities.pkl", "wb"))

        # save similarity matrix
        np.save("./fsSimMatrix.npy", fsSimMatrix)

    # Check if lavan similarity matrix exists
    if os.path.exists("./lavanSimMatrix.npy"):
        lavanSimMatrix = np.load("./lavanSimMatrix.npy")
        lavanIdentities = pickle.load(open("./lavanIdentities.pkl", "rb"))
    elif os.path.exists("./lavanReps.npy"):
        lavanReps = np.load("./lavanReps.npy")
        lavanIdentities = pickle.load(open("./lavanIdentities.pkl", "rb"))
        lavanSimMatrix = cosine_similarity(lavanReps)
        del lavanReps
        np.save("./lavanSimMatrix.npy", lavanSimMatrix)
    else:
        lavanFiles = [
            file for file in files if re.match(r"^\d", file) is not None
        ]

        # Get identities
        lavanIdentities = [name[:-4] for name in lavanFiles]

        # Combine files and identities
        lavanIdentities = pd.DataFrame(
            {"file": lavanFiles, "identity": lavanIdentities}
        )

        # Get Lavan reps
        lavanReps = get_reps(model, lavanFiles, stimDir)

        # Get similarity
        lavanSimMatrix = cosine_similarity(lavanReps)

        # Save reps
        np.save("./lavanReps.npy", lavanReps)
        del lavanReps

        # Save identity list
        pickle.dump(lavanIdentities, open("./lavanIdentities.pkl", "wb"))

        # save similarity matrix
        np.save("./lavanSimMatrix.npy", lavanSimMatrix)

    # Check if Mahnob similarity matrix exists
    if os.path.exists("./mahnobSimMatrix.npy"):
        mahnobSimMatrix = np.load("./mahnobSimMatrix.npy")
        mahnobIdentities = pickle.load(open("./mahnobIdentities.pkl", "rb"))
    elif os.path.exists("./mahnobReps.npy"):
        mahnobReps = np.load("./mahnobReps.npy")
        mahnobIdentities = pickle.load(open("./mahnobIdentities.pkl", "rb"))
        mahnobSimMatrix = cosine_similarity(mahnobReps)
        del mahnobReps
        np.save("./mahnobSimMatrix.npy", mahnobSimMatrix)
    else:
        mahnobFiles = [file for file in files if file[:3] == "sbj"]

        # Get identities
        mahnobIdentities = [
            "M" + re.search(r"(?<=sbj)\d*", file).group(0)
            for file in mahnobFiles
        ]

        # Combine files and identities
        mahnobIdentities = pd.DataFrame(
            {"file": mahnobFiles, "identity": mahnobIdentities}
        )

        # Get Mahnob reps
        mahnobReps = get_reps(model, mahnobFiles, stimDir)

        # Get similarity
        mahnobSimMatrix = cosine_similarity(mahnobReps)

        # Save reps
        np.save("./mahnobReps.npy", mahnobReps)
        del mahnobReps

        # Save identity list
        pickle.dump(mahnobIdentities, open("./mahnobIdentities.pkl", "wb"))

        # save similarity matrix
        np.save("./mahnobSimMatrix.npy", mahnobSimMatrix)

    # Loop through identities from high counts to low
    trialDf = pd.DataFrame(
        columns=[
            "TrialN",
            "Target",
            "Choice1",
            "Choice2",
            "CorrRes",
            "DiffScore",
        ]
    )

    # Make fs trials
    # Get identity counts of fs identities
    fsIdentityCounts = pd.Series(
        fsIdentities, index=fsIdentities
    ).value_counts()

    usedIndices = []
    # Loop until no identities have more than 1 file left
    while len(fsIdentityCounts[fsIdentityCounts > 1]) > 0:
        # Pick the identity with the most files
        identity = fsIdentityCounts.index[0]

        # Get indices of all files for this identity
        identityIndices = [
            i for i, x in enumerate(fsIdentities) if x == identity
        ]

        # Pick two files at random
        targetIndex, corrIndex = np.random.choice(identityIndices, 2)

        # Get the similarity between target and corr
        targetCorrSim = fsSimMatrix[targetIndex, corrIndex]

        # Select the file that is most dissimilar to the target
        foilIndices = np.argsort(fsSimMatrix[targetIndex, :])[::-1]

        for foil in foilIndices:
            if foil in identityIndices:
                continue
            else:
                foilIndex = foil
                break

        # Pick correct answer
        corrRes = np.random.choice([1, 2])
        # Fill in the trial dataframe
        trialDf = trialDf.append(
            {
                "TrialN": len(trialDf) + 1,
                "Target": fsFiles[targetIndex],
                "Choice1": fsFiles[corrIndex]
                if corrRes == 1
                else fsFiles[foilIndex],
                "Choice2": fsFiles[foilIndex]
                if corrRes == 1
                else fsFiles[corrIndex],
                "CorrRes": 1,
                "DiffScore": targetCorrSim
                - fsSimMatrix[targetIndex, foilIndex],
            },
            ignore_index=True,
        )

        # Remove the used files from identities and similarity matrix
        fsIdentities = [
            x
            for i, x in enumerate(fsIdentities)
            if i not in [targetIndex, corrIndex, foilIndex]
        ]
        fsSimMatrix = np.delete(
            np.delete(fsSimMatrix, [targetIndex, corrIndex, foilIndex], 0),
            [targetIndex, corrIndex, foilIndex],
            1,
        )

        # Recalculate identity counts
        fsIdentityCounts = pd.Series(
            fsIdentities, index=fsIdentities
        ).value_counts()
        print("stop")

    # # Keep looping until all identities have less than 1 files
    # usedFiles = []
    # repeatLimit = 1
    # negFirst = True
    # while len(identityCounts[identityCounts > 1]) > 0:
    #     # Pick the identity with the most files
    #     identity = identityCounts.index[0]

    #     # Get all files for this identity
    #     identityFiles = df[df["identity"] == identity]["file"].unique()

    #     # Check if this identity less than 2 files
    #     if len(identityFiles) < 2:
    #         continue

    #     # Pick two files at random
    #     targetFile = np.random.choice(identityFiles)
    #     nonTargetFile = np.random.choice(
    #         identityFiles[identityFiles != targetFile]
    #     )

    #     # Get their similarity scores
    #     targetRep = df[df["file"] == targetFile]["rep"].values[0]
    #     nonTargetRep = df[df["file"] == nonTargetFile]["rep"].values[0]
    #     corrSim = np.sum((targetRep - nonTargetRep) ** 2)

    #     # Get all other files that are not that identity
    #     foilFiles = df[df["identity"] != identity]["file"].unique()

    #     # Calculate similarity between the target and foils
    #     tmpDf = pd.DataFrame(columns=["file", "sim"])
    #     for foilFile in foilFiles:
    #         foilRep = df[df["file"] == foilFile]["rep"].values[0]
    #         foilSim = np.sum((targetRep - foilRep) ** 2)
    #         tmpDf = tmpDf.append(
    #             {"file": foilFile, "sim": foilSim}, ignore_index=True
    #         )

    #     # Pick the most (dis)similar file
    #     tmpDf = tmpDf.sort_values(by=["sim"], ascending=not negFirst)
    #     foilFile = tmpDf.iloc[-1]["file"]

    #     # Add files to used files
    #     usedFiles.append(targetFile)
    #     usedFiles.append(nonTargetFile)
    #     usedFiles.append(foilFile)

    #     # Remove used files from df if this is the second time to use this file
    #     if usedFiles.count(targetFile) >= repeatLimit:
    #         df = df[df["file"] != targetFile]

    #     if usedFiles.count(nonTargetFile) >= repeatLimit:
    #         df = df[df["file"] != nonTargetFile]

    #     if usedFiles.count(foilFile) >= repeatLimit:
    #         df = df[df["file"] != foilFile]

    #     # Recalculate counts
    #     identityCounts = df["identity"].value_counts()
    #     identityCounts = identityCounts.sort_values(ascending=False)

    #     # Pick correct response at random
    #     corrRes = np.random.choice([1, 2])

    #     # Add this row to the trials df
    #     trialDf = trialDf.append(
    #         {
    #             "Target": targetFile,
    #             "Choice1": nonTargetFile if corrRes == 1 else foilFile,
    #             "Choice2": foilFile if corrRes == 1 else nonTargetFile,
    #             "CorrRes": corrRes,
    #             "DiffScore": corrSim - tmpDf.iloc[-1]["sim"],
    #         },
    #         ignore_index=True,
    #     )

    # # Order trials by difficulty
    # trialDf = trialDf.sort_values(by=["DiffScore"], ascending=False)

    # # Add trial N
    # trialDf["TrialN"] = range(1, len(trialDf) + 1)

    # # Save as json
    # trialDf.to_json("./laughTrials.json", orient="records")
