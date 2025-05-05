import math
import random
import time
import os

# Global Parameters (set according to your dataset)
labLen = 140           # Number of labeled users (e.g., 1%)
unLen = 3408          # Total number of users
MAXEDGES = 3000000    # Maximum number of network edges
ClassNum = 2

spamclass = 1         # Increase the weight of spam class
DD = 1                # Weight d default (e.g., 1)
LL = 0                # Weight Lambda
MAXITTOTAL = 10       # Maximum total iterations
THRESHOLD = 0.5       # Spam probability threshold for crisp label assignment

# Global arrays
# For ease of matching the C++ indexing (1-indexed), index 0 will be unused.
UnUPM = [None] * (unLen + 1)        # List of UPM objects, index 1..unLen
UnLabels = [-10] * (unLen + 1)        # Ground truth labels for each user
Train_Index = [0] * labLen            # Stores user IDs for training examples
Train_Label = [0] * labLen            # Stores labels for training examples
Predit_Pro = [0.0] * (unLen + 1)        # Spam probabilities from logistic regression
alpha_k = [0.0] * ClassNum            # Parameters alpha for each class

datasetForLR = []   # List of LRUP objects (generated for logistic regression)

# ----------------- Data Structure Classes -----------------

class UPM:
    def __init__(self):
        self.uID = 0
        self.shill = -10      # -10 indicates uninitialized (or not part of dataset)
        self.tempLab = -10
        self.neighbors = []   # list of neighbor user IDs
        # Instead of a fixed-size array, use lists:
        self.pTheta = [0.0] * ClassNum         # estimated probability for each class
        self.pFinalLabelWeight = [0.0] * ClassNum
        # For each neighbor, we store a list of ClassNum floats (z values)
        self.z_jk = []  # will be built as a list with length == len(neighbors)

class LRUP:
    def __init__(self, Data_Index=0, tmpLab=0, w_jk_i=0.0):
        self.Data_Index = Data_Index  # user id
        self.tmpLab = tmpLab          # label (0 or 1)
        self.w_jk_i = w_jk_i          # instance weight

# ----------------- Function: Python_LR_Invoke -----------------
# This function calls a logistic regression routine implemented in a separate module.
# You must have a module named 'python2c' with a function 'LR_First' that accepts three lists:
#   LR_First(Index, pLabel, iWeight) -> (resIndex, resSpamPro)
def python_lr_invoke(Index, pLabel, iWeight):
    try:
        import python2c  # your own module for LR
    except ImportError:
        print("Module python2c not found!")
        return -1
    
    # print(Index)
    ret = python2c.LR_First(Index, pLabel, iWeight)
    if ret is None:
        return 0
    resIndex, resSpamPro = ret
    if len(resIndex) != len(resSpamPro):
        print("Return Results Error!")
        return 0
    # Update global Predit_Pro
    global Predit_Pro
    Predit_Pro = [0.0] * (unLen + 1)
    for j in range(len(resIndex)):
        tIndex = resIndex[j]
        tProb = resSpamPro[j]
        if tIndex < len(Predit_Pro):
            Predit_Pro[tIndex] = tProb
    return 1

# ----------------- Function: initialization -----------------
def initialization():
    global UnUPM, UnLabels, Train_Index, Train_Label

    # Initialize UnUPM array (index 1..unLen)
    for i in range(1, unLen + 1):
        UnUPM[i] = UPM()
        UnUPM[i].shill = -10
        UnLabels[i] = -10

    # Read training file
    # train_path = os.path.join("dataset", "5percent", "train_4.csv")
    train_path = "/data1/jianweiw/LLM/Imputation/Fake_review_detection/Fake_Review_Detection/Group_Spammer/datasets/he_amazon/Training_Testing/5percent/train_4.csv"
    try:
        with open(train_path, "r") as fin1:
            train_lines = fin1.readlines()
    except IOError:
        print(f"Failed to open {train_path}")
        return

    t_idx = 0
    for line in train_lines:
        # Assuming each line contains two numbers separated by whitespace (userID and labelID)
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        userID = int(parts[0])
        labelID = int(parts[1])
        if userID > unLen:
            continue
        UnUPM[userID].shill = labelID
        UnUPM[userID].tempLab = labelID
        UnUPM[userID].uID = userID
        UnLabels[userID] = labelID
        if t_idx < labLen:
            Train_Index[t_idx] = userID
            Train_Label[t_idx] = labelID
            t_idx += 1
            # print(labelID)

    # Read test file
    # test_path = os.path.join("dataset", "5percent", "test_4.csv")
    test_path = "/data1/jianweiw/LLM/Imputation/Fake_review_detection/Fake_Review_Detection/Group_Spammer/datasets/he_amazon/Training_Testing/5percent/test_4.csv"
    try:
        with open(test_path, "r") as fin4:
            test_lines = fin4.readlines()
    except IOError:
        print(f"Failed to open {test_path}")
        return

    for line in test_lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        userID = int(parts[0])
        labelID = int(parts[1])
        if userID > unLen:
            continue
        UnUPM[userID].shill = -1
        UnUPM[userID].tempLab = -1
        UnUPM[userID].uID = userID
        UnLabels[userID] = labelID

    # Read neighbor file (assumes file is sorted by userID)
    # neighbor_path = os.path.join("dataset", "jaccard0.2.txt")
    neighbor_path = "/data1/jianweiw/LLM/Imputation/Fake_review_detection/Fake_Review_Detection/Group_Spammer/datasets/he_amazon/J01Network.txt"
    try:
        with open(neighbor_path, "r") as fin3:
            neighbor_lines = fin3.readlines()
    except IOError:
        print(f"Failed to open {neighbor_path}")
        return

    tempUserID = None
    cNeighbors = 0
    for line in neighbor_lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        userID = int(parts[0])
        neighbor = int(parts[1])
        if tempUserID is None:
            tempUserID = userID
        if userID == tempUserID:
            UnUPM[tempUserID].neighbors.append(neighbor)
            cNeighbors += 1
        else:
            # Finished reading one user's neighbors; initialize the z_jk list with zeros
            UnUPM[tempUserID].z_jk = [[0.0]*ClassNum for _ in range(cNeighbors)]
            tempUserID = userID
            cNeighbors = 0
            UnUPM[tempUserID].neighbors.append(neighbor)
            cNeighbors += 1
    # For the last user:
    if tempUserID is not None:
        UnUPM[tempUserID].z_jk = [[0.0]*ClassNum for _ in range(cNeighbors)]

    # Compute network statistics
    total_edges = 0
    N0 = 0
    N1 = 0
    N00 = 0
    N11 = 0
    for i in range(1, unLen + 1):
        num_neighbors = len(UnUPM[i].neighbors)
        total_edges += num_neighbors
        if UnLabels[i] == 1:
            N1 += num_neighbors
        elif UnLabels[i] == 0:
            N0 += num_neighbors
        for neigh in UnUPM[i].neighbors:
            if UnLabels[i] == 1 and UnLabels[neigh] == 1:
                N11 += 1
            if UnLabels[i] == 0 and UnLabels[neigh] == 0:
                N00 += 1
    print("Number of Edges in Network:", total_edges,
          "Total Edges of Normal:", N0,
          "Total Edges of Spammer:", N1)
    print("Number of Spam-Spam Edges in Network:", N11,
          "Number of Normal-Normal Edges in Network:", N00)
    if N1 != 0 and N0 != 0:
        print("Purity1:", N11 / N1, "  Purity0:", N00 / N0)

# ----------------- Function: LossFunction -----------------
def LossFunction(lambd, d):
    Loss = 0.0
    for i in range(1, unLen + 1):
        user = UnUPM[i]
        # Skip if user not properly initialized
        if user is None:
            continue
        # Use a minimum value to avoid log(0)
        P1 = user.pTheta[1] if user.pTheta[1] >= 1e-6 else 1e-6
        P0 = user.pTheta[0] if user.pTheta[0] >= 1e-6 else 1e-6

        if user.shill == -1 and user.tempLab == 1:
            Loss -= lambd * math.log(P1)
        elif user.shill == 1:
            Loss -= math.log(P1)
        elif user.shill == -1 and user.tempLab == 0:
            Loss -= lambd * math.log(P0)
        elif user.shill == 0:
            Loss -= math.log(P0)

        if user.shill != -10 and len(user.neighbors) > 0:
            ww = 1.0
            PP = 0.0
            if user.shill == -1:
                ww = lambd
            if user.shill == 1 or user.tempLab == 1:
                for idx, neigh_id in enumerate(user.neighbors):
                    neighbor = UnUPM[neigh_id]
                    if neighbor.shill == 1:
                        PP += math.log(alpha_k[1])
                    elif neighbor.shill == 0:
                        PP += math.log(alpha_k[0])
                    elif neighbor.shill == -1:
                        NP1 = neighbor.pTheta[1] if neighbor.pTheta[1] >= 1e-6 else 1e-6
                        NP0 = neighbor.pTheta[0] if neighbor.pTheta[0] >= 1e-6 else 1e-6
                        if neighbor.tempLab == 1:
                            PP += math.log(alpha_k[1] * NP1)
                        else:
                            PP += math.log(alpha_k[0] * NP0)
            if user.shill == 0 or user.tempLab == 0:
                for idx, neigh_id in enumerate(user.neighbors):
                    neighbor = UnUPM[neigh_id]
                    if neighbor.shill == 1:
                        PP += math.log(1 - alpha_k[1])
                    elif neighbor.shill == 0:
                        PP += math.log(1 - alpha_k[0])
                    elif neighbor.shill == -1:
                        NP1 = neighbor.pTheta[1] if neighbor.pTheta[1] >= 1e-6 else 1e-6
                        NP0 = neighbor.pTheta[0] if neighbor.pTheta[0] >= 1e-6 else 1e-6
                        if neighbor.tempLab == 1:
                            PP += math.log((1 - alpha_k[1]) * NP1)
                        else:
                            PP += math.log((1 - alpha_k[0]) * NP0)
            Loss -= (d * ww / len(user.neighbors)) * PP
    print("Go out loss function....")
    return Loss

# ----------------- Function: Random_Noise -----------------
def Random_Noise():
    # Generate a noise term similar to the C++ version
    rr = random.random()
    rrr = (rr - 0.5) * 0.3
    return rrr

# ----------------- Function: InitClassifier -----------------
def InitClassifier(lambd):
    # Initialize instance weights for training instances (all ones)
    iWeight = [1.0] * labLen
    # Invoke logistic regression on the training data
    python_lr_invoke(Train_Index, Train_Label, iWeight)
    
    # Update pTheta and pFinalLabelWeight for each user
    for i in range(1, unLen + 1):
        user = UnUPM[i]
        if user.shill == 1:
            user.pTheta[1] = 1.0
            user.pTheta[0] = 0.0
        elif user.shill == 0:
            user.pTheta[1] = 0.0
            user.pTheta[0] = 1.0
        elif user.shill == -1:
            user.pTheta[1] = Predit_Pro[i]
            user.pTheta[0] = 1.0 - Predit_Pro[i]
        # Copy probabilities to final label weights
        user.pFinalLabelWeight[0] = user.pTheta[0]
        user.pFinalLabelWeight[1] = user.pTheta[1]
        # Set temporary label based on threshold
        user.tempLab = 1 if user.pTheta[1] >= THRESHOLD else 0

    # Compute the numerator for alpha_k using neighbors from labeled users
    clsFriendsNum = [0.0, 0.0]
    for i in range(1, unLen + 1):
        user = UnUPM[i]
        if user.shill != -10 and len(user.neighbors) > 0:
            if user.shill == 1:
                for neigh_id in user.neighbors:
                    neighbor = UnUPM[neigh_id]
                    if neighbor.shill == 1:
                        clsFriendsNum[1] += 1
                    elif neighbor.shill == 0:
                        clsFriendsNum[0] += 1
                    elif neighbor.shill == -1:
                        if neighbor.tempLab == 1:
                            clsFriendsNum[1] += 1
                        else:
                            clsFriendsNum[0] += 1
    # Compute and print performance on unlabeled users
    tp = fn = fp = tn = 0
    for i in range(1, unLen + 1):
        user = UnUPM[i]
        if user.shill == -1:
            if user.tempLab == 1 and UnLabels[i] == 1:
                tp += 1
            if user.tempLab == 0 and UnLabels[i] == 1:
                fn += 1
            if user.tempLab == 1 and UnLabels[i] == 0:
                fp += 1
            if user.tempLab == 0 and UnLabels[i] == 0:
                tn += 1
    print("tp =", tp)
    print("fn =", fn)
    print("fp =", fp)
    print("tn =", tn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f_measure = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    print("RECALL =", recall)
    print("PRECISION =", precision)
    print("F-MEASURE =", f_measure)
    # Compute alpha_k using the counts from neighbors
    s = clsFriendsNum[0] + clsFriendsNum[1]
    if s != 0:
        alpha_k[0] = clsFriendsNum[0] / s
        alpha_k[1] = clsFriendsNum[1] / s
    print("alpha[0] =", alpha_k[0])
    print("alpha[1] =", alpha_k[1])
    Loss1 = LossFunction(lambd, DD)
    print("InitLR Loss:", Loss1)

# ----------------- Function: ComputeAlphaK -----------------
def ComputeAlphaK(lambd, d):
    clsFriendsNum = [0.0, 0.0]
    allFriendsNum = [0.0, 0.0]
    for i in range(1, unLen + 1):
        user = UnUPM[i]
        if user.shill != -10 and len(user.neighbors) > 0:
            for idx, neigh_id in enumerate(user.neighbors):
                neighbor = UnUPM[neigh_id]
                if user.shill == 1 or user.shill == 0:
                    if user.shill == 1:
                        clsFriendsNum[1] += user.z_jk[idx][1] / len(user.neighbors)
                        clsFriendsNum[0] += user.z_jk[idx][0] / len(user.neighbors)
                        allFriendsNum[0] += user.z_jk[idx][0] / len(user.neighbors)
                        allFriendsNum[1] += user.z_jk[idx][1] / len(user.neighbors)
                    else:
                        allFriendsNum[0] += user.z_jk[idx][0] / len(user.neighbors)
                        allFriendsNum[1] += user.z_jk[idx][1] / len(user.neighbors)
                else:
                    if neighbor.tempLab == 1:
                        clsFriendsNum[1] += (lambd * user.z_jk[idx][1]) / len(user.neighbors)
                        clsFriendsNum[0] += (lambd * user.z_jk[idx][0]) / len(user.neighbors)
                        allFriendsNum[0] += (lambd * user.z_jk[idx][0]) / len(user.neighbors)
                        allFriendsNum[1] += (lambd * user.z_jk[idx][1]) / len(user.neighbors)
                    else:
                        allFriendsNum[0] += (lambd * user.z_jk[idx][0]) / len(user.neighbors)
                        allFriendsNum[1] += (lambd * user.z_jk[idx][1]) / len(user.neighbors)
    for k in range(ClassNum):
        if allFriendsNum[k] != 0:
            alpha_k[k] = clsFriendsNum[k] / allFriendsNum[k]
        else:
            alpha_k[k] = 0.0
    print("alpha[0] =", alpha_k[0])
    print("alpha[1] =", alpha_k[1])

# ----------------- Function: EStep_Crisp -----------------
def EStep_Crisp(lambd, d):
    print("Go into EStep....")
    for i in range(1, unLen + 1):
        user = UnUPM[i]
        if user.shill != -10:
            # Reset z_jk to have the same length as neighbors
            if len(user.z_jk) != len(user.neighbors):
                user.z_jk = [[0.0]*ClassNum for _ in range(len(user.neighbors))]
            for idx, neigh_id in enumerate(user.neighbors):
                neighbor = UnUPM[neigh_id]
                if neighbor.shill == 0:
                    user.z_jk[idx] = [1.0, 0.0]
                elif neighbor.shill == 1:
                    user.z_jk[idx] = [0.0, 1.0]
                elif neighbor.shill == -1:
                    if user.shill == 0 or user.tempLab == 0:
                        # Compute normalized weights for each class using (1-alpha_k)
                        NP0 = neighbor.pTheta[0] if neighbor.pTheta[0] >= 1e-6 else 1e-6
                        NP1 = neighbor.pTheta[1] if neighbor.pTheta[1] >= 1e-6 else 1e-6
                        summ = (1 - alpha_k[0]) * NP0 + (1 - alpha_k[1]) * NP1
                        user.z_jk[idx][0] = ((1 - alpha_k[0]) * NP0) / summ if summ != 0 else 0
                        user.z_jk[idx][1] = ((1 - alpha_k[1]) * NP1) / summ if summ != 0 else 0
                    elif user.shill == 1 or user.tempLab == 1:
                        NP0 = neighbor.pTheta[0] if neighbor.pTheta[0] >= 1e-6 else 1e-6
                        NP1 = neighbor.pTheta[1] if neighbor.pTheta[1] >= 1e-6 else 1e-6
                        summ = alpha_k[0] * NP0 + alpha_k[1] * NP1
                        user.z_jk[idx][0] = (alpha_k[0] * NP0) / summ if summ != 0 else 0
                        user.z_jk[idx][1] = (alpha_k[1] * NP1) / summ if summ != 0 else 0
    ComputeAlphaK(lambd, d)

# ----------------- Function: IterLogReg -----------------
def IterLogReg(lambd, d):
    global datasetForLR
    print("Start Generating New Dataset....")
    datasetForLR = []  # clear previous dataset
    cc = 0  # counter for datasetForLR entries
    # Insert nodes themselves
    for i in range(1, unLen + 1):
        user = UnUPM[i]
        if user.shill != -10:
            inst = LRUP()
            if user.shill == -1 and lambd > 0:
                inst.w_jk_i = lambd
            elif user.shill in [0, 1]:
                inst.w_jk_i = 1.0
            inst.tmpLab = user.tempLab
            inst.Data_Index = user.uID
            datasetForLR.append(inst)
            cc += 1
    print("After Inserting Nodes ITSELVES, Data Size:", cc)
    # Insert neighbor instances if conditions are met
    for i in range(1, unLen + 1):
        user = UnUPM[i]
        if user.shill != -10 and len(user.neighbors) > 0 and (user.pTheta[1] >= 0.9 or user.pTheta[1] <= 0.01):
            for idx, neigh_id in enumerate(user.neighbors):
                neighbor = UnUPM[neigh_id]
                if neighbor.shill == -1:
                    for k in range(ClassNum):
                        inst = LRUP()
                        if user.shill in [0, 1]:
                            inst.w_jk_i = DD * user.z_jk[idx][k] / len(user.neighbors)
                        else:
                            inst.w_jk_i = lambd * DD * user.z_jk[idx][k] / len(user.neighbors)
                        inst.tmpLab = k
                        inst.Data_Index = neighbor.uID
                        datasetForLR.append(inst)
                        cc += 1
                if neighbor.shill == 0:
                    inst = LRUP()
                    if user.shill in [0, 1]:
                        inst.w_jk_i = DD / len(user.neighbors)
                    else:
                        inst.w_jk_i = lambd * DD / len(user.neighbors)
                    inst.tmpLab = 0
                    inst.Data_Index = neighbor.uID
                    datasetForLR.append(inst)
                    cc += 1
                if neighbor.shill == 1:
                    inst = LRUP()
                    if user.shill in [0, 1]:
                        inst.w_jk_i = DD / len(user.neighbors)
                    else:
                        inst.w_jk_i = lambd * DD / len(user.neighbors)
                    inst.tmpLab = 1
                    inst.Data_Index = neighbor.uID
                    datasetForLR.append(inst)
                    cc += 1
    print("Size of New DATASETS:", cc)
    # Prepare lists for logistic regression
    tTrainIndex = []
    tTrainLabel = []
    tIWeight = []
    for inst in datasetForLR:
        tTrainIndex.append(inst.Data_Index)
        tTrainLabel.append(inst.tmpLab)
        tIWeight.append(inst.w_jk_i)
    python_lr_invoke(tTrainIndex, tTrainLabel, tIWeight)

# ----------------- Function: Select_Top_Neighbor -----------------
def Select_Top_Neighbor(K, index):
    # Returns a list of indices (within the neighbor list) for the top K neighbors.
    top = []
    currentMin = 1.0
    MinIndex = 0
    k = 0
    user = UnUPM[index]
    for j in range(len(user.neighbors)):
        if k < K:
            top.append(j)
            k += 1
        else:
            currentMin = 1.0
            for p in range(K):
                neigh = UnUPM[user.neighbors[top[p]]]
                if neigh.pTheta[1] < currentMin:
                    currentMin = neigh.pTheta[1]
                    MinIndex = p
            if UnUPM[user.neighbors[j]].pTheta[1] > currentMin:
                top[MinIndex] = j
    return top

# ----------------- Function: MStep -----------------
def MStep(lambd, d):
    print("Start MStep ...")
    IterLogReg(lambd, d)
    # Update probabilities for each user
    for i in range(1, unLen + 1):
        user = UnUPM[i]
        if user.shill == 1:
            user.pTheta[1] = 1.0
            user.pTheta[0] = 0.0
        elif user.shill == 0:
            user.pTheta[1] = 0.0
            user.pTheta[0] = 1.0
        elif user.shill == -1:
            user.pTheta[1] = Predit_Pro[i]
            user.pTheta[0] = 1.0 - Predit_Pro[i]
            if len(user.neighbors) > 10 and user.pTheta[1] < 0.4:
                user.pTheta[1] += 0.2
                user.pTheta[0] = 1.0 - user.pTheta[1]
    # Update pTheta using neighbors’ information
    for i in range(1, unLen + 1):
        user = UnUPM[i]
        if user.shill == -1 and len(user.neighbors) > 0:
            # Initialize mul with current probabilities (avoid zeros)
            mul = [val if val > 0 else 0.0001 for val in user.pTheta]
            neighW = [1.0, 1.0]
            for neigh_id in user.neighbors:
                neighbor = UnUPM[neigh_id]
                if neighbor.shill == -1:
                    tempSum0 = (1 - alpha_k[0]) * neighbor.pTheta[0] + (1 - alpha_k[1]) * neighbor.pTheta[1]
                    tempSum0 = math.pow(tempSum0, d / len(user.neighbors))
                    neighW[0] *= tempSum0
                    tempSum1 = alpha_k[0] * neighbor.pTheta[0] + alpha_k[1] * neighbor.pTheta[1]
                    tempSum1 = math.pow(tempSum1, d / len(user.neighbors))
                    neighW[1] *= tempSum1
                if neighbor.shill == 0:
                    neighW[0] *= math.pow(1 - alpha_k[0], d / len(user.neighbors))
                    neighW[1] *= math.pow(alpha_k[0], d / len(user.neighbors))
                if neighbor.shill == 1:
                    neighW[1] *= math.pow(alpha_k[1], d / len(user.neighbors))
                    neighW[0] *= math.pow(1 - alpha_k[1], d / len(user.neighbors))
            for k in range(ClassNum):
                mul[k] *= neighW[k]
            summ = mul[0] + mul[1]
            if summ != 0:
                mul[0] /= summ
                mul[1] /= summ
            user.pTheta = mul[:]
            user.pFinalLabelWeight = mul[:]
            user.tempLab = 1 if mul[1] >= THRESHOLD else 0

    Loss = LossFunction(lambd, d)
    # Write predictions to file and compute performance metrics
    # output_path = os.path.join("spammer_results", "prediction_1percents.txt")
    output_path = "./our_output_files/AmazonLRF"
    tp = fn = fp = tn = 0
    try:
        with open(output_path, "w") as fout:
            for i in range(1, unLen + 1):
                user = UnUPM[i]
                if user.shill == -1:
                    if user.tempLab == 1 and UnLabels[i] == 1:
                        tp += 1
                    if user.tempLab == 0 and UnLabels[i] == 1:
                        fn += 1
                    if user.tempLab == 1 and UnLabels[i] == 0:
                        fp += 1
                    if user.tempLab == 0 and UnLabels[i] == 0:
                        tn += 1
                    fout.write(f"{i} {user.tempLab}\n")
    except IOError:
        print(f"Could not write to {output_path}")
    print("tp =", tp, ", Spammer to Spammer")
    print("fn =", fn, ", Spammer to Normal")
    print("fp =", fp, ", Normal to Spammer")
    print("tn =", tn, ", Normal to Normal")
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f_measure = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    print("RECALL =", recall)
    print("PRECISION =", precision)
    print("F-MEASURE =", f_measure)
    return Loss

# ----------------- Function: Output_to_File -----------------
def Output_to_File(k):
    file_name = f"./our_output_files/AmazonLRF{k}"
    try:
        with open(file_name, "w") as fout:
            for i in range(1, unLen + 1):
                user = UnUPM[i]
                if user.shill == -1:
                    fout.write(f"{i},{user.pTheta[1]}\n")
    except IOError:
        print(f"Could not write to {file_name}")

def SaveResults():
    output_path = "../detection_results/he_amazon/isr_predictions.txt"
    tp = fn = fp = tn = 0
    with open(output_path, "w") as fout:
        fout.write("sample_index\tprediction\n")
        for i in range(1, unLen + 1):
            user = UnUPM[i]
            if user.shill == -1:
                if user.tempLab == 1 and UnLabels[i] == 1:
                    tp += 1
                if user.tempLab == 0 and UnLabels[i] == 1:
                    fn += 1
                if user.tempLab == 1 and UnLabels[i] == 0:
                    fp += 1
                if user.tempLab == 0 and UnLabels[i] == 0:
                    tn += 1
                fout.write(f"{i}\t{user.tempLab}\n")
    
    output_path_metrics = "../detection_results/he_amazon/isr_metrics.txt"

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f_measure = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

    with open(output_path_metrics, 'w') as f_out:
        f_out.write(f"F-measure: {f_measure:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n")
    print(f"Saved evaluation metrics to {output_path_metrics}")

# ----------------- Function: control -----------------
def control(lambd, d):
    initialization()
    print("The lambda is:", lambd)
    InitClassifier(lambd)
    IterNum = 0
    Loss = 0.0
    while IterNum < MAXITTOTAL:
        EStep_Crisp(lambd, d)
        Loss = MStep(lambd, d)
        IterNum += 1
        print("Iteration Number:", IterNum, ", Loss:", Loss)
        Output_to_File(IterNum)
        SaveResults()

# ----------------- Main -----------------
if __name__ == "__main__":
    control(LL, DD)
