import pandas as pd
import numpy as np
import os
import re


def trialMaker(identities, clips, sims, laughV1Trials, laughV2Trials):
    identityCounts = identities["identity"].value_counts()
    newTrials = pd.DataFrame(columns=["Target", "Choice", "Foil", "DiffScore"])
    while np.any(identityCounts > 1) and len(identityCounts) > 1:
        # Get all identities with more than 1 identity
        viableIdentities = identityCounts.loc[
            identityCounts > np.min(identityCounts)
        ].index

        possibleTrials = pd.DataFrame(columns=["Target", "Choice", "Foil", "DiffScore"])
        for identity in viableIdentities:
            # Get all clips with this identity
            targetClips = identities.loc[
                identities["identity"] == identity, "file"
            ].values

            # Get id for each of these clips
            targetIDs = identities.loc[
                identities["file"].isin(targetClips), "id"
            ].values

            # Get sims for each of these clips
            targetSims = sims[targetIDs, :]
            targetSims = targetSims[:, targetIDs]

            # Get the most dissimilar pair
            pairIdx = np.unravel_index(np.argmax(targetSims), targetSims.shape)
            targetClip1, targetClip2 = [targetIDs[x] for x in pairIdx]

            # Get the target similarity for this pair
            targetSim = targetSims[pairIdx]

            # Get all foil identities based on what's left
            foilIdentities = identityCounts.loc[
                identityCounts == np.min(identityCounts)
            ].index

            # Make trials where clip1 is target
            for singleIdentity in foilIdentities:
                # Get all clips with this identity
                foilClips = identities.loc[
                    identities["identity"] == singleIdentity, "file"
                ].values

                # Get id for each of these clips
                foilClipID = identities.loc[
                    identities["file"].isin(foilClips), "id"
                ].values

                # Get sims for each of these clips
                foilClipSim = sims[targetClip1, foilClipID[0]]

                # Add to possibleTrials
                possibleTrials = pd.concat(
                    [
                        possibleTrials,
                        pd.DataFrame(
                            {
                                "Target": identities.loc[targetClip1, "file"],
                                "Choice": identities.loc[targetClip2, "file"],
                                "Foil": identities.loc[foilClipID, "file"],
                                "DiffScore": targetSim - foilClipSim,
                            }
                        ),
                    ]
                )

            # Make trials where clip2 is target
            for singleIdentity in foilIdentities:
                # Get all clips with this identity
                foilClips = identities.loc[
                    identities["identity"] == singleIdentity, "file"
                ].values

                # Get id for each of these clips
                foilClipID = identities.loc[
                    identities["file"].isin(foilClips), "id"
                ].values

                # Get sims for each of these clips
                foilClipSim = sims[targetClip2, foilClipID[0]]

                # Add to possibleTrials
                possibleTrials = pd.concat(
                    [
                        possibleTrials,
                        pd.DataFrame(
                            {
                                "Target": identities.loc[targetClip2, "file"],
                                "Choice": identities.loc[targetClip1, "file"],
                                "Foil": identities.loc[foilClipID, "file"],
                                "DiffScore": targetSim - foilClipSim,
                            }
                        ),
                    ]
                )

        # Sort trials by difficulty
        possibleTrials = possibleTrials.sort_values("DiffScore", ascending=False)

        while True:
            # Pick the most difficult trial
            trial = possibleTrials.iloc[0]

            # Check if it exists V1 trials
            if trial["Target"] in laughV1Trials["Target"].values:
                # Get v1 trial
                v1Trial = laughV1Trials.loc[laughV1Trials["Target"] == trial["Target"]]

                # Get clips from V1 trial
                corChoice = v1Trial["CorrRes"].values[0]
                foilChoice = 2 if corChoice == 1 else 1
                v1CorChoice = v1Trial["Choice" + str(corChoice)].values
                v2FoilChoice = v1Trial["Choice" + str(foilChoice)].values

                # Check if V1 clips are the same as the proposed clip
                if v1CorChoice == trial["Choice"] and v2FoilChoice == trial["Foil"]:
                    # Remove trial from possibleTrials
                    possibleTrials = possibleTrials.iloc[1:]
                    continue

            # Check if it exists in V2 trials
            if trial["Target"] in laughV2Trials["Target"].values:
                # Get v2 trial
                v2Trial = laughV2Trials.loc[laughV2Trials["Target"] == trial["Target"]]

                # Get clips from V2 trial
                corChoice = v2Trial["CorrRes"].values[0]
                foilChoice = 2 if corChoice == 1 else 1
                v2CorChoice = v2Trial["Choice" + str(corChoice)].values
                v2FoilChoice = v2Trial["Choice" + str(foilChoice)].values

                # Check if V2 clips are the same as the proposed clip
                if v2CorChoice == trial["Choice"] and v2FoilChoice == trial["Foil"]:
                    # Remove trial from possibleTrials
                    possibleTrials = possibleTrials.iloc[1:]
                    continue

            break

        # Add trial to newTrials
        newTrials = pd.concat(
            [
                newTrials,
                pd.DataFrame(
                    {
                        "Target": [trial["Target"]],
                        "Choice": [trial["Choice"]],
                        "Foil": [trial["Foil"]],
                        "DiffScore": [trial["DiffScore"]],
                    }
                ),
            ]
        )

        # Remove clips from clips
        clips.remove(trial["Target"])
        clips.remove(trial["Choice"])
        clips.remove(trial["Foil"])

        # Only keep identities that are in the unused clips
        identities = identities.loc[identities["file"].isin(clips)]

        # Recalculate identityCounts
        identityCounts = identities["identity"].value_counts()

    return newTrials


if __name__ == "__main__":
    # Load V1 trial json
    laughV1Trials = pd.read_json("laughTrialsV1.json")

    laughSummary = pd.read_csv("laughTrialSummary.csv")

    # Get all files used in current trials
    laughClips = laughSummary[["Target", "Choice1", "Choice2"]]

    # Flatten the list
    laughClips = laughClips.values.flatten()

    # List files in stimuli folder
    stimuli = os.listdir("stimuli")

    # Get all files that are not used in current trials
    unusedClips = [x for x in stimuli if x not in laughClips]

    # Fill in V1Cor with Cor if V1Cor is -1
    laughSummary.loc[laughSummary["V1Cor"] == -1, "V1Cor"] = laughSummary.loc[
        laughSummary["V1Cor"] == -1, "Cor"
    ]

    # Remap CorrRes column from f and j to 1 and 2
    laughSummary.loc[laughSummary["CorrRes"] == "f", "CorrRes"] = 1
    laughSummary.loc[laughSummary["CorrRes"] == "j", "CorrRes"] = 2

    # Calculate mean between Cor and V1Cor
    laughSummary["MeanCor"] = laughSummary[["Cor", "V1Cor"]].mean(axis=1)

    # Get bad trials
    badTrials = laughSummary.loc[
        laughSummary["MeanCor"] < 0.1,
    ]

    # Split up unused clips into sources
    fsClips = [x for x in unusedClips if x.startswith("FS")]
    mahnobClips = [x for x in unusedClips if x.startswith("sbj")]
    pattern = re.compile(r"^[0-9]+")
    lavanClips = [x for x in unusedClips if re.match(pattern, x) is not None]

    # Create new trials with fsClips
    # Load fs information
    fsIdentities = pd.read_pickle("fsIdentities.pkl")
    fsIdentities["id"] = range(len(fsIdentities))

    # Only keep identities that are in the unused clips
    fsIdentities = fsIdentities.loc[fsIdentities["file"].isin(fsClips)]

    fsSims = np.load("fsSimMatrix.npy")

    newTrials = trialMaker(fsIdentities, fsClips, fsSims, laughV1Trials, laughSummary)

    # Create new trials with Mahnob clips
    mahnobIdentities = pd.read_pickle("mahnobIdentities.pkl")
    mahnobSims = np.load("mahnobSimMatrix.npy")
    mahnobIdentities["id"] = range(len(mahnobIdentities))

    # Only keep identities that are in the unused clips
    mahnobIdentities = mahnobIdentities.loc[mahnobIdentities["file"].isin(mahnobClips)]

    newTrials = pd.concat(
        [
            newTrials,
            trialMaker(
                mahnobIdentities, mahnobClips, mahnobSims, laughV1Trials, laughSummary
            ),
        ]
    )

    # Create new trials with Lavan clips
    lavanIdentities = pd.read_pickle("lavanIdentities.pkl")
    lavanSims = np.load("lavanSimMatrix.npy")
    lavanIdentities["id"] = range(len(lavanIdentities))

    # Only keep identities that are in the unused clips
    lavanIdentities = lavanIdentities.loc[lavanIdentities["file"].isin(lavanClips)]

    newTrials = pd.concat(
        [
            newTrials,
            trialMaker(
                lavanIdentities, lavanClips, lavanSims, laughV1Trials, laughSummary
            ),
        ]
    )

    # Sort trials
    newTrials = newTrials.sort_values(by="DiffScore", ascending=False)

    # Reset index
    newTrials = newTrials.reset_index(drop=True)

    # Save new trials
    newTrials.to_csv("newTrialsForV3.csv", index=False)
    print("stop")
