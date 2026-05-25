import torch


def inject_edge_embeddings(
    model,
    input_ids: torch.Tensor,
    edge_item_ids,
    projector,
    movie_token_id: int,
    sanitize_projected: bool = False,
):
    device = input_ids.device
    inputs_embeds = model.get_input_embeddings()(input_ids).clone()

    movie_positions = input_ids == movie_token_id
    if not movie_positions.any():
        return inputs_embeds

    movie_indices = torch.where(movie_positions)
    unique_batch_indices = torch.unique(movie_indices[0])

    for batch_idx in unique_batch_indices:
        batch_movie_indices = movie_indices[1][movie_indices[0] == batch_idx]
        edge_ids = edge_item_ids[batch_idx] if edge_item_ids is not None else []

        if len(edge_ids) != len(batch_movie_indices):
            continue

        for movie_idx, edge_id in zip(batch_movie_indices, edge_ids):
            if edge_id not in projector.edge_embeddings:
                continue
            edge_emb = projector.edge_embeddings[edge_id].clone().detach().to(device)
            projected = projector(edge_emb.unsqueeze(0)).squeeze(0)
            if sanitize_projected:
                projected = torch.nan_to_num(projected, nan=0.0)
            inputs_embeds[batch_idx, movie_idx] = projected

    return inputs_embeds