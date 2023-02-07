import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import audioIO
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import pickle


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

    # Check if df pickle exists
    if os.path.exists("./laughDf.pkl"):
        # Load
        df = pd.read_pickle("./laughDf.pkl")
    else:
        stimDir = "./stimuli/"
        files = audioIO.get_files(stimDir)
        model = hub.load("https://tfhub.dev/google/trillsson2/1")

        # Filter FS files
        fsFiles = [file for file in files if file[:2] == "FS"]
        # Check if reps exist
        if os.path.exists("./fsReps.npy"):
            reps = np.load("./fsReps.npy")
            identities = pickle.load(open("./fsIdentities.pkl", "rb"))
        else:
            identities = get_FS_identities(fsFiles)

            reps = get_reps(model, fsFiles, stimDir)
            # Save reps
            np.save("./fsReps.npy", reps)
            # Save identity list
            pickle.dump(identities, open("./fsIdentities.pkl", "wb"))

        df = pd.DataFrame(
            data={
                "file": fsFiles,
                "identity": identities,
                "rep": [rep for rep in reps],
                "group": "FS",
            }
        )

        # Filter Lavan identities
        lavanFiles = [
            file for file in files if re.match(r"^\d", file) is not None
        ]
        # Check if reps exist
        if os.path.exists("./lavanReps.npy"):
            reps = np.load("./lavanReps.npy")
            identities = pickle.load(open("./lavanIdentities.pkl", "rb"))
        else:
            identities = ["L" + re.split(r"_", file)[0] for file in lavanFiles]

            reps = get_reps(model, lavanFiles, stimDir)
            # Save reps
            np.save("./lavanReps.npy", reps)
            # Save identity list
            pickle.dump(identities, open("./lavanIdentities.pkl", "wb"))

        # Attach lavan identities to df
        df = df.append(
            pd.DataFrame(
                data={
                    "file": lavanFiles,
                    "identity": identities,
                    "rep": [rep for rep in reps],
                    "group": "Lavan",
                }
            ),
            ignore_index=True,
        )

        # Filter mahnob files
        mahnobFiles = [file for file in files if file[:3] == "sbj"]

        # Check if reps exist
        if os.path.exists("./mahnobReps.npy"):
            reps = np.load("./mahnobReps.npy")
            identities = pickle.load(open("./mahnobIdentities.pkl", "rb"))
        else:
            identities = [
                "M" + re.search(r"(?<=sbj)\d*", file).group(0)
                for file in mahnobFiles
            ]

            reps = get_reps(model, mahnobFiles, stimDir)
            # Save reps
            np.save("./mahnobReps.npy", reps)
            # Save identity list
            pickle.dump(identities, open("./mahnobIdentities.pkl", "wb"))

        # Attach mahnob identities to df
        df = df.append(
            pd.DataFrame(
                data={
                    "file": mahnobFiles,
                    "identity": identities,
                    "rep": [rep for rep in reps],
                    "group": "Mahnob",
                }
            )
        )

        # # Filter Adrienne identities
        # adrienneFiles = [
        #     file for file in files if re.match(r"^(session)", file) is not None
        # ]
        # # Check if reps exist
        # if os.path.exists("./adrienneReps.npy"):
        #     reps = np.load("./adrienneReps.npy")
        #     identities = pickle.load(open("./adrienneIdentities.pkl", "rb"))
        # else:
        #     identities = [
        #         "A" + re.search(r"(?<=participant)\d*", file).group(0)
        #         for file in adrienneFiles
        #     ]
        #     reps = get_reps(model, adrienneFiles, stimDir)
        #     # Save reps
        #     np.save("./adrienneReps.npy", reps)
        #     # Save identity list
        #     pickle.dump(identities, open("./adrienneIdentities.pkl", "wb"))

        # Sort df by identity
        df = df.sort_values(by=["identity"])

        # Pickle df
        df.to_pickle("./laughDf.pkl")

    simMatrix = create_sim_matrix(df)
    plot_sim_matrix(simMatrix, "./laughSimMatrix.png")

    # Get identities and counts for each identity
    identities = df["identity"].unique()
    identityCounts = df["identity"].value_counts()
    identityCounts = identityCounts.sort_values(ascending=False)

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

    # Keep looping until all identities have less than 1 files
    usedFiles = []
    repeatLimit = 1
    negFirst = True
    while len(identityCounts[identityCounts > 1]) > 0:
        # Pick the identity with the most files
        identity = identityCounts.index[0]

        # Get all files for this identity
        identityFiles = df[df["identity"] == identity]["file"].unique()

        # Check if this identity less than 2 files
        if len(identityFiles) < 2:
            continue

        # Pick two files at random
        targetFile = np.random.choice(identityFiles)
        nonTargetFile = np.random.choice(
            identityFiles[identityFiles != targetFile]
        )

        # Get their similarity scores
        targetRep = df[df["file"] == targetFile]["rep"].values[0]
        nonTargetRep = df[df["file"] == nonTargetFile]["rep"].values[0]
        corrSim = np.sum((targetRep - nonTargetRep) ** 2)

        # Get all other files that are not that identity
        foilFiles = df[df["identity"] != identity]["file"].unique()

        # Calculate similarity between the target and foils
        tmpDf = pd.DataFrame(columns=["file", "sim"])
        for foilFile in foilFiles:
            foilRep = df[df["file"] == foilFile]["rep"].values[0]
            foilSim = np.sum((targetRep - foilRep) ** 2)
            tmpDf = tmpDf.append(
                {"file": foilFile, "sim": foilSim}, ignore_index=True
            )

        # Pick the most (dis)similar file
        tmpDf = tmpDf.sort_values(by=["sim"], ascending=not negFirst)
        foilFile = tmpDf.iloc[-1]["file"]

        # Add files to used files
        usedFiles.append(targetFile)
        usedFiles.append(nonTargetFile)
        usedFiles.append(foilFile)

        # Remove used files from df if this is the second time to use this file
        if usedFiles.count(targetFile) >= repeatLimit:
            df = df[df["file"] != targetFile]

        if usedFiles.count(nonTargetFile) >= repeatLimit:
            df = df[df["file"] != nonTargetFile]

        if usedFiles.count(foilFile) >= repeatLimit:
            df = df[df["file"] != foilFile]

        # Recalculate counts
        identityCounts = df["identity"].value_counts()
        identityCounts = identityCounts.sort_values(ascending=False)

        # Pick correct response at random
        corrRes = np.random.choice([1, 2])

        # Add this row to the trials df
        trialDf = trialDf.append(
            {
                "Target": targetFile,
                "Choice1": nonTargetFile if corrRes == 1 else foilFile,
                "Choice2": foilFile if corrRes == 1 else nonTargetFile,
                "CorrRes": corrRes,
                "DiffScore": corrSim - tmpDf.iloc[-1]["sim"],
            },
            ignore_index=True,
        )

    # Order trials by difficulty
    trialDf = trialDf.sort_values(by=["DiffScore"], ascending=False)

    # Add trial N
    trialDf["TrialN"] = range(1, len(trialDf) + 1)

    # Save as json
    trialDf.to_json("./laughTrials.json", orient="records")
