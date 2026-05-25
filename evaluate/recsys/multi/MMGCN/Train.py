from tqdm import tqdm


def train(epoch, length, dataloader, model, optimizer, batch_size, writer=None):
    model.train()
    print('Now, training start ...')
    sum_loss = 0.0
    sum_model_loss = 0.0
    sum_reg_loss = 0.0
    sum_ent_loss = 0.0
    sum_weight_loss = 0.0
    step = 0.0
    pbar = tqdm(total=length)
    num_pbar = 0
    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()
        loss, reg_loss, bpr_loss, embedding_loss, pref_loss = model.loss(user_tensor, item_tensor)
        loss.backward()
        optimizer.step()
        pbar.update(batch_size)
        num_pbar += batch_size
        sum_loss += loss.cpu().item()
        sum_reg_loss += reg_loss.cpu().item()
        sum_model_loss += bpr_loss.cpu().item()
        sum_weight_loss += embedding_loss.cpu().item()
        sum_ent_loss += pref_loss.cpu().item()
        step += 1.0
    pbar.close()
    print('----------------- total_loss:{}  bpr_loss:{}  regularization_loss:{} preference_reg:{} embedding_reg:{}--------------'
        .format(sum_loss/step, sum_model_loss/step, sum_reg_loss/step, sum_ent_loss/step, sum_weight_loss/step))
    if writer is not None:
        writer.add_scalar('loss', sum_loss/step, epoch)
        writer.add_scalar('model_loss', sum_model_loss/step, epoch)
        writer.add_scalar('reg_loss', sum_reg_loss/step, epoch)

    return loss
