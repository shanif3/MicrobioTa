# src/training/utils.py
import torch

def embedding_regularization(embeddings, taxonomy_distance_matrix, mode, num_of_partition):
    num_bacteria, embedding_dim = embeddings.shape
    if mode == 'exponent_on_h':
        squared_distances = torch.cdist(embeddings, embeddings, p=2) ** 2
        reg_matrix = torch.exp(-taxonomy_distance_matrix.float()) * squared_distances
        reg_loss = torch.sum(reg_matrix.abs()) / (num_bacteria * num_bacteria)
        return reg_loss * num_of_partition


def adjust_embedding_for_test_lodo(attn_mask,abundances, name_embeddings, batch_size, bacteria_names_train, bacteria_names_test,
                                   device_to_use, microbes_to_mask=None, microbes_to_mask_index=None):
    train_name_to_index = {name: i for i, name in enumerate(bacteria_names_train)}
    test_name_to_index = {name: i for i, name in enumerate(bacteria_names_test)}


    # Handle masking case
    if microbes_to_mask is not None:
        microbes_not_in_mask = list(set(bacteria_names_train) - set(microbes_to_mask))
        microbes_not_in_mask_indices = [train_name_to_index[name] for name in microbes_not_in_mask]
        taxonomy_matrix = calc_distance_in_sample(bacteria_names_train, device_to_use)

        # Process masked bacteria in a batch if possible
        for masked_bac_index in microbes_to_mask_index:
            values = taxonomy_matrix[masked_bac_index].float()
            values[microbes_to_mask_index] = float('inf')


            # Initialize average tensor
            average = torch.zeros((batch_size, name_embeddings.shape[2]), device=device_to_use)
            min_indices= torch.min(taxonomy_matrix[masked_bac_index].float(), dim=2)[0]
            if min_indices.numel() > 0:
                average=torch.mean(name_embeddings[:,min_indices,:], dim=1)
            # if (values == 3.0).any():
            #     min_index = torch.nonzero(values == 3.0, as_tuple=True)[0][0]
            #     average = name_embeddings[:, min_index, :]
            # elif (values == 2.0).any():
            #     min_indices = torch.nonzero(values == 2.0, as_tuple=True)[0]
            #     average= torch.mean(name_embeddings[:, min_indices, :],dim=1)
            #
            # elif (values == 4.0).any():
            #     attn_mask[:, masked_bac_index] = 1

            # Apply the average only once
            name_embeddings[:, masked_bac_index] = average * abundances[:, masked_bac_index].unsqueeze(-1)


        # Process non-masked bacteria using tensor operations if possible
        for not_masked_bac in microbes_not_in_mask:
            train_index = train_name_to_index[not_masked_bac]
            name_embeddings[:, train_index] = name_embeddings[:, train_index] * abundances[:, train_index].unsqueeze(-1)

        return name_embeddings, attn_mask

    # Non-masking case (test mode)
    # Create mapping from test to train indices
    test_to_train_indices = {}
    # Calculate taxonomy matrix once at the beginning if needed
    if microbes_to_mask is not None or len(set(bacteria_names_test) - set(bacteria_names_train)) > 0:
        merge_bact_names = list(dict.fromkeys(bacteria_names_train + bacteria_names_test))
        taxonomy_matrix = calc_distance_in_sample(merge_bact_names, device_to_use)

    for i, name in enumerate(bacteria_names_train):
        if name in bacteria_names_test:
            test_to_train_indices[i] = test_name_to_index[name]

    # Zero out bacteria not in test set and scale others by abundance
    mask = torch.zeros(len(bacteria_names_train), device=device_to_use, dtype=torch.bool)
    for train_idx, test_idx in test_to_train_indices.items():
        mask[train_idx] = True
        name_embeddings[:, train_idx, :] *= abundances[:, test_idx].unsqueeze(-1)

    # Zero out the embeddings for bacteria not in the test set
    name_embeddings[:, ~mask, :] = 0

    # Handle bacteria in test but not in train
    diff_bacteria = list(set(bacteria_names_test) - set(bacteria_names_train))
    if not diff_bacteria:
        return name_embeddings

    # Extend embeddings tensor for new bacteria
    x = torch.cat((
        name_embeddings,
        torch.zeros(batch_size, len(diff_bacteria), name_embeddings.shape[2], device=device_to_use)
    ), dim=1)

    attn_mask = torch.zeros(batch_size, x.shape[1], dtype=torch.bool, device=device_to_use)
    additional_mask_for_extending_x= torch.ones(len(diff_bacteria), dtype=torch.bool, device=device_to_use) # True for new bacteria- not padded
    mask= torch.cat((mask,additional_mask_for_extending_x))
    attn_mask[:, ~mask] = 1


    # Process each unique bacterium not in training set
    taxonomy_subset = taxonomy_matrix[:, :len(bacteria_names_train)]

    for i, unique_bac in enumerate(diff_bacteria):
        bact_index = merge_bact_names.index(unique_bac)
        values = taxonomy_subset[bact_index]

        average = torch.zeros((batch_size, x.shape[2]), device=device_to_use)
        average = torch.zeros((batch_size, name_embeddings.shape[2]), device=device_to_use)
        min_indices = torch.min(taxonomy_matrix[bact_index].float(), dim=2)[0]
        if min_indices.numel() > 0:
            average = torch.mean(name_embeddings[:, min_indices, :], dim=1)

        # if (values == 3).any():
        #     idx = torch.nonzero(values == 3, as_tuple=True)[0][0]
        #     average = x[:, idx, :]
        # elif (values == 2.0).any():
        #     min_indices = torch.nonzero(values == 2.0, as_tuple=True)[0]
        #     average = name_embeddings[:, min_indices, :].sum(dim=1) / len(min_indices)
        # elif (values == 4.0).any():
        #     attn_mask[:, bact_index] = 1

        test_index = test_name_to_index[unique_bac]
        x[:, len(bacteria_names_train) + i] = average * abundances[:, test_index].unsqueeze(-1)

    return x,attn_mask



def calc_distance_in_sample(bact_names, device):
    # Pre-split each bacteria name
    split_names = [tuple(name.split(';')) for name in bact_names]
    N = len(split_names)

    # Initialize result tensor directly on the specified device
    C = torch.zeros((N, N), dtype=torch.uint8, device=device)

    # Calculate upper triangle indices directly on the device
    row_idx, col_idx = torch.triu_indices(N, N, 1, device=device)

    # Calculate distances using list comprehension and transfer results to device
    distances = [
        calc_dist_pair_split(split_names[i], split_names[j])
        for i, j in zip(row_idx.tolist(), col_idx.tolist())
    ]
    C[row_idx, col_idx] = torch.tensor(distances, dtype=torch.uint8, device=device)

    # Mirror upper triangle to lower triangle to make C symmetric
    C += C.T.clone()

    return C

def calc_dist_pair_split(l1, l2) -> int:
    if l1 == l2:
        return 0  # exact match

    len_l1, len_l2 = len(l1), len(l2)
    min_len = min(len_l1, len_l2)

    for k, (s1, s2) in enumerate(zip(l1[:min_len], l2[:min_len])):
        if s1 != s2:
            return (len_l1 - k) + (len_l2 - k)
    return abs(len_l1 - len_l2)
    # # Find where they first differ
    # mismatch_idx = next((k for k in range(min_len) if l1[k] != l2[k]), min_len)
    #
    # # Case: sibling species (share same parent level, then differ)
    # if mismatch_idx == 2 and (l1[2]!='XXX' and  l2[2]!='XXX'):
    #     return 2
    #
    # # Case: one is parent/ancestor of the other
    # if mismatch_idx == 2 and (l1[2] == 'XXX' or l2[2] == 'XXX'):
    #     return 3
    #
    # # Default: partial match but not parent-child or sibling
    # return mismatch_idx