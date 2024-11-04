#@title wenhong
# wenhong
import random
import csv
from collections import Counter

f = open("records.csv", "r")
records = {}
count = 0
no_of_members = 5

headers = list(f.readline().rstrip("\n").split(","))

# from str to one-hot or vice versa
def convert_school(sch, val):
  if type(val) == str:
    sch_map =  [0] * len(sch)
    sch_map[sch.index(val)] = 1
    return sch_map
  else:
    return sch[val.index(1)]

def convert_gender(val):
  if type(val) == str:
    if val == "Male":
      return [1,0]
    else:
      return [0,1]
  else:
    if val == [1,0]:
      return "Male"
    else:
      return "Female"

# return x groups of k
def number_of_groups(total_no_students, k):
  '''
  Description:
  This function provides a dictionary of x number of groups containing k members.

  Example: 50, 5
  Returns {10:5}
  '''
  if total_no_students % k == 0:
    return {total_no_students//k:k}
  else:
    lesser_groups = k - total_no_students % k
    return{lesser_groups: k-1, int((total_no_students-lesser_groups*(k-1))/5):k}

tutorial_groups = {}
lines = f.readlines()
for line in lines:
  # if count == 50:
  #   break
  line = line.rstrip("\n").split(",")
  if line[0] not in tutorial_groups.keys():
    tutorial_groups[line[0]] = {}
    tutorial_groups[line[0]]["records"] = [line]
    tutorial_groups[line[0]]["schools"] = [line[2]]
  else:
    tutorial_groups[line[0]]["records"].append(line)
    if line[2] not in tutorial_groups[line[0]]["schools"]:
      tutorial_groups[line[0]]["schools"].append(line[2])
  # count += 1

# convert schools and gender to one-hot

for group in tutorial_groups.keys():
  for rec in tutorial_groups[group]["records"]:
    rec[2] = convert_school(tutorial_groups[group]["schools"], rec[2])
    rec[4] = convert_gender(rec[4])
    rec.append(rec[2]+rec[4]+[float(rec[5])])
  # print(tutorial_groups[group]["schools"])

# Custom function to calculate the K-Prototypes distance between two points
def k_prototype_distance(point, medoid, categorical_indices):
    # Numerical distance using squared euclidean distance
    num_distance = (point[-1] - medoid[-1]) ** 2
    # Categorical distance using matching dissimilarity
    # sch_cat = categorical_indices[:-2]
    # gen_cat = categorical_indices[-2:]
    # cat_distance_sch = sum((1/len(sch_cat)) for i in sch_cat if point[i] != centroid[i])
    # cat_distance_gen = sum((1/len(gen_cat)) for i in gen_cat if point[i] != centroid[i])
    # cat_distance = cat_distance_sch + cat_distance_gen
    cat_distance = sum(1 for i in categorical_indices if point[i] != medoid[i]) / len(categorical_indices)
    combined_distance = num_distance + cat_distance
    return combined_distance

# # Cao initialization function
# def cao_init(data, categorical_indices, k):
#   centroids = []
#   density = [0] * len(data)

#   # Calculate density for each data point based on categorical values
#   for idx in categorical_indices:
#     # Count occurrences of each value in the categorical feature
#     value_counts = Counter(row[idx] for row in data)
#     # Add frequency to each point's density
#     for i, row in enumerate(data):
#       density[i] += value_counts[row[idx]]

#   # Choose first centroid as the point with highest density
#   first_centroid_idx = density.index(max(density))
#   centroids.append(data[first_centroid_idx])

#   # Select remaining centroids based on max distance from existing centroids
#   for _ in range(1, k):
#     max_dist_idx = -1
#     max_distance = -1
#     for i, row in enumerate(data):
#       if row not in centroids:
#           # Compute the minimum distance from the point to any existing centroid
#         min_dist = min(k_prototype_distance(row, centroid, categorical_indices) for centroid in centroids)
#         if min_dist > max_distance:
#           max_distance = min_dist
#           max_dist_idx = i
#     centroids.append(data[max_dist_idx])

#   return centroids

# K-Prototypes-Medoids clustering function
def k_prototypes_medoids(data, categorical_indices, k, max_iter=100):
  # Step 1: Initialize medoids using random sample
  medoids = random.sample(data, k)

  # Iterative optimization process
  for _ in range(max_iter):
    # Step 2: Assign points to clusters based on nearest medoid
    clusters = [[] for _ in range(k)]
    for point in data:
      distances = [k_prototype_distance(point, medoid, categorical_indices) for medoid in medoids]
      cluster_idx = distances.index(min(distances))
      clusters[cluster_idx].append(point)

    # Step 3: Update medoids by choosing the most central point in each cluster
    new_medoids = []
    for cluster in clusters:
      if cluster:
        # For each point in the cluster, calculate the sum distance to all other points in the cluster
        medoid_distances = []
        for candidate in cluster:
          total_distance = sum(k_prototype_distance(candidate, other, categorical_indices) for other in cluster)
          medoid_distances.append(total_distance)

        # Choose the point with minimum total distance as the new medoid
        min_distance_idx = medoid_distances.index(min(medoid_distances))
        new_medoids.append(cluster[min_distance_idx])
      else:
        # Reinitialize empty cluster medoid randomly
        new_medoids.append(random.choice(data))

    # Check for convergence
    if new_medoids == medoids:
      break
    for i in range(len(new_medoids)-1):
      for j in range(i+1, len(new_medoids)):
        if new_medoids[i] == new_medoids[j]:
          
          print(f"MEDOIDS {new_medoids} {new_medoids[i]} {new_medoids[j]}")
    medoids = new_medoids

  return medoids

def calculate_combined_variance(group, categorical_indices, numeric_index):
    """
    Calculate a combined variance measure for a group with both numeric and categorical data.

    Parameters:
    - group: List of data points in the cluster, where each point is a list of features.
    - categorical_indices: List of indices corresponding to categorical features.
    - numeric_indices: List of indices corresponding to numeric features.

    Returns:
    - combined_variance: Combined variance measure for the group.
    """

    # Calculate mean for the numeric feature
    mean = sum(point[numeric_index] for point in group) / len(group)
    # Calculate variance for the numeric feature
    numeric_variance = sum((point[numeric_index] - mean) ** 2 for point in group) / len(group)
    # print(f"Numeric variance for {group}: {numeric_variance}")

    # Step 2: Calculate categorical dispersion
    categorical_dispersions = []
    for index in categorical_indices:
      # Calculate the mode frequency for this categorical feature
      values = [point[index] for point in group]
      frequency_counts = Counter(values)
      mode_value, mode_count = frequency_counts.most_common(1)[0]

      # Calculate dispersion as the proportion of points that are NOT the mode
      dispersion = 1 - (mode_count / len(group))
      categorical_dispersions.append(dispersion)

    # Calculate average categorical dispersion
    avg_categorical_dispersion = sum(categorical_dispersions) / len(categorical_dispersions) * 10
    # Step 3: Combine numeric and categorical measures
    combined_variance = numeric_variance + avg_categorical_dispersion
    # print(numeric_variance, avg_categorical_dispersion, combined_variance)
    # print(f"Categorical variance: {avg_categorical_dispersion}")
    # print(combined_variance)
    return combined_variance

def z_score_normalize(data):
    mean_val = sum(data) / len(data)
    variance = sum((x - mean_val) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    normalized_data = [(x - mean_val) / std_dev for x in data]
    return normalized_data

def max_dist_pair(point, other_group, categorical_indices):
  max_distance = 0
  pair = []
  for other in other_group:
    curr_distance = k_prototype_distance(point, other, categorical_indices)
    if curr_distance > max_distance:
      max_distance = curr_distance
      pair = [point, other]
  return max_distance, pair

def point_swap(group1, group2, categorical_indices, curr_variance, mean_variance):
  n = len(group1) // 2
  
  potential_pairs = {}
  largest = {}
  for point in group1:
    max_distance_pair = max_dist_pair(point, group2, categorical_indices)
    potential_pairs[max_distance_pair[0]] = max_distance_pair[1]
    
  greatest_distance = max(potential_pairs)
  largest[greatest_distance] = potential_pairs[greatest_distance]
  # print(largest)
  for distance in largest.keys():
    new_group_1 = [i for i in group1 if i != largest[distance][0]] + [largest[distance][1]]
    # print([i[-1] for i in new_group_1])
    var_group_1 = calculate_combined_variance(new_group_1, categorical_indices, len(categorical_indices))
    new_group_2 = [i for i in group2 if i != largest[distance][1]] + [largest[distance][0]]
    var_group_2 = calculate_combined_variance(new_group_2, categorical_indices, len(categorical_indices))
    if var_group_1 > curr_variance and var_group_2 >= mean_variance:
      # print(new_group_1[-1][-1], group1[0][-1])
      # print(new_group_2[-1][-1], group2[0][-1])
      largest[distance][0][-1], largest[distance][1][-1] = largest[distance][1][-1], largest[distance][0][-1]
  #     print("SWAPPED!")
  # print(f"new group 1: {[i[-1] for i in group1]}")
  # print(f"new group 2: {[i[-1] for i in group2]}")
      return group1.index(largest[distance][0]), group2.index(largest[distance][1]), 1
  return 0,0,0

# Define the dataset
for group in tutorial_groups.keys():

  data = [rec[-1] for rec in tutorial_groups[group]["records"]]
  data_copy = data.copy()
  # all_numeric_data = [row[-1] for row in data]

  # normalized_numeric = z_score_normalize(all_numeric_data)
  # for i,row in enumerate(data):
  #   row[-1] = normalized_numeric[i]

  # Categorical indices (based on data structure)
  categorical_indices = [i for i in range(len(data[0])-1)]

  no_groups = number_of_groups(len(data), no_of_members)
  current_group_no = 1
  current_group = []
  for no_sub_group in no_groups.keys():
    k = no_groups[no_sub_group]
    total_combined_variance = 0
    for _ in range(no_sub_group):
      # Run the K-Prototypes-Medoids algorithm
      medoids = k_prototypes_medoids(data, categorical_indices, k)
      for medoid in medoids:
        # if medoid not in data:
        #   print(medoid)
        data.remove(medoid)
        medoid.append(current_group_no)
        current_group.append(medoid)
        # finalised_data.append(medoid)
      current_group_no += 1
        # Calculate combined variance for the group
      combined_variance = calculate_combined_variance(current_group, categorical_indices, len(categorical_indices))

      total_combined_variance += combined_variance
      current_group = []
      # print(len(data))
    # print("Mean Combined Variance for the tutorial group:", total_combined_variance/sum(i for i in no_groups))
  # optimise the formed groups
  total_no_groups = sum(_ for _ in no_groups)
  mean_variance = total_combined_variance/total_no_groups
  grouped_data = [[] for _ in range(total_no_groups)]
  
  for assigned_student in data_copy:
    if type(assigned_student[-1]) is not int:
      x = convert_school(tutorial_groups[group]["schools"], assigned_student[:len(categorical_indices)])
      print(group, x, assigned_student)
    # grouped_data[assigned_student[-1]-1].append(assigned_student)
  # for formed_group in grouped_data:
    
  #   filtered_group = [grp for grp in grouped_data if grp != formed_group]
  #   # find distance of n longest distance points
  #   curr_variance = calculate_combined_variance(formed_group, categorical_indices, len(categorical_indices))
  #   if curr_variance >= mean_variance:
  #     continue
    # for group in filtered_group:
    #   pt_group_1, pt_group_2, swapped = point_swap(formed_group, group, categorical_indices, curr_variance, mean_variance)
    #   if swapped:
    #     formed_group[pt_group_1], group[pt_group_2] = group[pt_group_2], formed_group[pt_group_1]
      # print([i[-1] for i in formed_group])

def MyFunc(e):
  return e[-1][-1]

new_headers = headers
new_headers.append("Team Assigned")
with open('new_records.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(new_headers)
    for group in tutorial_groups.keys():

      tutorial_groups[group]["records"].sort(key=MyFunc)
      for rec in tutorial_groups[group]["records"]:
        rec[2] = convert_school(tutorial_groups[group]["schools"], rec[2])
        rec[4] = convert_gender(rec[4])
        rec[-1] = rec[-1][-1]
        writer.writerow(rec)


f.close()