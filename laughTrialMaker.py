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


def create_trials(identities, simMatrix):
    # Loop through identities from high counts to low
    trialDf = pd.DataFrame(
        columns=[
            "Target",
            "Choice1",
            "Choice2",
            "CorrRes",
            "DiffScore",
        ]
    )

    # Make fs trials
    # Get identity counts of fs identities
    identityCounts = identities["identity"].value_counts()

    usedIndices = []
    # Loop until no identities have more than 1 file left
    while len(identityCounts[identityCounts > 1]) > 0:
        # Pick the identity with the most files
        identity = identityCounts.index[0]

        # Get indices of all files for this identity
        identityIndices = identities[identities["identity"] == identity].index

        # Pick two files at random
        targetIndex, corrIndex = np.random.choice(identityIndices, 2)

        # Get the similarity between target and corr
        targetCorrSim = simMatrix[targetIndex, corrIndex]

        # Select the file that is most dissimilar to the target
        foilIndices = np.argsort(simMatrix[targetIndex, :])[::-1]

        for foil in foilIndices:
            if foil in identityIndices:
                continue
            else:
                foilIndex = foil
                break

        # Pick correct answer
        corrRes = np.random.choice([1, 2])
        # Fill in the trial dataframe
        trialDf = pd.concat(
            [
                trialDf,
                pd.DataFrame(
                    {
                        "Target": [identities["file"][targetIndex]],
                        "Choice1": [identities["file"][corrIndex]]
                        if corrRes == 1
                        else [identities["file"][foilIndex]],
                        "Choice2": [identities["file"][foilIndex]]
                        if corrRes == 1
                        else [identities["file"][corrIndex]],
                        "CorrRes": [corrRes],
                        "DiffScore": [
                            targetCorrSim - simMatrix[targetIndex, foilIndex]
                        ],
                    }
                ),
            ]
        )

        # Remove the used files from identities and similarity matrix
        identities = identities.drop([targetIndex, corrIndex, foilIndex])
        identities = identities.reset_index(drop=True)
        simMatrix = np.delete(
            np.delete(simMatrix, [targetIndex, corrIndex, foilIndex], 0),
            [targetIndex, corrIndex, foilIndex],
            1,
        )

        # Recalculate identity counts
        identityCounts = identities["identity"].value_counts()

    return trialDf


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
        lavanIdentities = [
            name.split("_")[-1].split(".")[0] for name in lavanFiles
        ]

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

    # Make trials
    trialDf = create_trials(fsIdentities, fsSimMatrix)
    trialDf["source"] = "joanne"

    tmp = create_trials(lavanIdentities, lavanSimMatrix)
    tmp["source"] = "lavan"
    trialDf = pd.concat([trialDf, tmp])

    tmp = create_trials(mahnobIdentities, mahnobSimMatrix)
    tmp["source"] = "mahnob"
    trialDf = pd.concat([trialDf, tmp])

    # Sort trials based on difficulty
    trialDf = trialDf.sort_values("DiffScore", ascending=False)

    # Save trials as json
    trialDf.to_json("./laughTrials.json", orient="records")
