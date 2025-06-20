import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import wandb

def Successive_Record_Steps_COS_Similarity(recorded_steps_top_eigenvectors):
    Successive_Record_Steps_COS_Similarity = []
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
    num_steps = len(sorted_steps)

    for i in range(num_steps):
        current_step = sorted_steps[i]
        current_eigenvector = recorded_steps_top_eigenvectors[current_step][:, 0].reshape(1, -1)

        if i > 0:
            previous_step = sorted_steps[i - 1]
            previous_eigenvector = recorded_steps_top_eigenvectors[previous_step][:, 0].reshape(1, -1)
            cos_sim = abs(cosine_similarity(current_eigenvector, previous_eigenvector)[0][0])
            Successive_Record_Steps_COS_Similarity.append((current_step, cos_sim))
            wandb.log({"Successive_Record_Steps_COS_Similarity": cos_sim}, step=current_step)
        elif i == 0:
            Successive_Record_Steps_COS_Similarity.append((current_step, np.nan)) # 对于第一个步骤，没有前一个，所以相似度为 NaN
            wandb.log({"Successive_Record_Steps_COS_Similarity": np.nan}, step=current_step)

    return Successive_Record_Steps_COS_Similarity

def First_Last_Record_Steps_COS_Similarity(recorded_steps_top_eigenvectors):
    First_Last_Record_Steps_COS_Similarity = []
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
    num_steps = len(sorted_steps)

    if num_steps < 2:
        return First_Last_Record_Steps_COS_Similarity # 如果记录的步骤少于两个，则返回空列表

    last_step = sorted_steps[-1]
    Last_max_eigenvector = recorded_steps_top_eigenvectors[last_step][:, 0].reshape(1, -1)

    for i in range(num_steps - 1):
        current_step = sorted_steps[i]
        current_eigenvector = recorded_steps_top_eigenvectors[current_step][:, 0].reshape(1, -1)
        cos_sim = abs(cosine_similarity(current_eigenvector, Last_max_eigenvector)[0][0])
        First_Last_Record_Steps_COS_Similarity.append((current_step, cos_sim))
        wandb.log({"First_Last_Record_Steps_COS_Similarity": cos_sim}, step=current_step)

    return First_Last_Record_Steps_COS_Similarity