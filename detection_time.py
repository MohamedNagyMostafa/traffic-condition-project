from cv2 import sort
import pandas as pd

preds_path = '/home/manuel/Documents/khalifa/Msc Thesis/codes/traffic_project/ls_lf_best_bi_lstm_v2.csv'
ground_truth_path = '/home/manuel/Documents/khalifa/Msc Thesis/codes/traffic_project/test_data2.csv'


preds = pd.read_csv(preds_path, header=None).to_numpy()
ground_truth = pd.read_csv(ground_truth_path, header=None).to_numpy()


print(ground_truth.shape)

preds_idxs = []
ground_truth_idxs = []

preds_on_stream = False
ground_truth_on_stream = False

for idx in range(len(preds)):
    if preds[idx] == 0:
        preds_on_stream = False
    elif not preds_on_stream:
        preds_on_stream = True
        preds_idxs.append(idx)

    if ground_truth[idx, 1] == 0:
        ground_truth_on_stream = False
    elif not ground_truth_on_stream:
        ground_truth_on_stream = True
        ground_truth_idxs.append(idx)


print(preds_idxs)
print(ground_truth_idxs)


# Get the nearest prediction before or after the ground truth

total_detection_time = 0
closest_distance = 100000000
closest_idx = 0

differences = []

for idx in ground_truth_idxs:
    for idx2 in preds_idxs:
        if abs(idx - idx2) < closest_distance:
            closest_idx = idx2
            closest_distance = abs(idx - idx2)

    print(idx, closest_idx)

    diff = ground_truth[closest_idx, 0] - ground_truth[idx, 0]
    differences.append(diff)
    
    total_detection_time += diff

    closest_distance = 1000000000

# mean
print(total_detection_time / len(ground_truth_idxs))

# median
differences.sort()
print(differences[len(differences) // 2])

