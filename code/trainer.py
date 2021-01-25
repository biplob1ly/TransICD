import torch
import logging
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
import torch.optim as optim
from run_manager import RunManager
from torch.utils.data import DataLoader


def train(model, train_set, dev_set, test_set, hyper_params, batch_size, device):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    m = RunManager()
    optimizer = optim.AdamW(model.parameters(), lr=hyper_params.learning_rate)

    logging.info("Training Started...")
    m.begin_run(hyper_params, model, train_loader)
    for epoch in range(hyper_params.num_epoch):
        m.begin_epoch(epoch + 1)
        model.train()
        for batch in train_loader:
            texts = batch['text']
            lens = batch['length']
            targets = batch['codes']

            texts = texts.to(device)
            targets = targets.to(device)
            outputs, ldam_outputs, _ = model(texts, targets)

            if ldam_outputs is not None:
                loss = F.binary_cross_entropy_with_logits(ldam_outputs, targets)
            else:
                loss = F.binary_cross_entropy_with_logits(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m.track_loss(loss)
            # m.track_num_correct(preds, affinities)

        m.end_epoch()
    m.end_run()
    hype = '_'.join([f'{k}_{v}' for k, v in hyper_params._asdict().items()])
    m.save(f'../results/train_results_{hype}')
    logging.info("Training finished.\n")

    # Training
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    probabs, targets, _, _ = evaluate(model, train_loader, device, dtset='train')
    compute_scores(probabs, targets, hyper_params, dtset='train')

    # Validation
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=1)
    probabs, targets, _, _ = evaluate(model, dev_loader, device, dtset='dev')
    compute_scores(probabs, targets, hyper_params, dtset='dev')

    # test_dataset
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)
    probabs, targets, full_hadm_ids, full_attn_weights = evaluate(model, test_loader, device, dtset='test')
    compute_scores(probabs, targets, hyper_params, dtset='test', full_hadm_ids=full_hadm_ids, full_attn_weights=full_attn_weights)


def evaluate(model, loader, device, dtset):
    fin_targets = []
    fin_probabs = []
    full_hadm_ids = []
    full_attn_weights = []

    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        for batch in loader:
            hadm_ids = batch['hadm_id']
            texts = batch['text']
            lens = batch['length']
            targets = batch['codes']

            texts = texts.to(device)
            targets = targets
            outputs, _, attn_weights = model(texts)

            fin_targets.extend(targets.tolist())
            fin_probabs.extend(torch.sigmoid(outputs).detach().cpu().tolist())
            if dtset == 'test' and attn_weights is not None:
                full_hadm_ids.extend(hadm_ids)
                full_attn_weights.extend(attn_weights.detach().cpu().tolist())
    return fin_probabs, fin_targets, full_hadm_ids, full_attn_weights


def save_predictions(probabs, targets, dtset, hype):
    np.savetxt(f'../results/{dtset}_probabs_{hype}.txt', probabs)
    np.savetxt(f'../results/{dtset}_targets_{hype}.txt', targets)


def precision_at_k(true_labels, pred_probs):
    # num true labels in top k predictions / k
    ks = [1, 5, 8, 10, 15]
    sorted_pred = np.argsort(pred_probs)[:, ::-1]
    output = []
    p5_scores = None
    for k in ks:
        topk = sorted_pred[:, :k]

        # get precision at k for each example
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = true_labels[i, tk].sum()
                denom = len(tk)
                vals.append(num_true_in_top_k / float(denom))

        output.append(np.mean(vals))
        if k == 5:
            p5_scores = np.array(vals)
    return output, p5_scores


def compute_scores(probabs, targets, hyper_params, dtset, full_hadm_ids=None, full_attn_weights=None):
    probabs = np.array(probabs)
    targets = np.array(targets)

    preds = np.rint(probabs)  # (probabs >= 0.5)
    accuracy = metrics.accuracy_score(targets, preds)
    f1_score_micro = metrics.f1_score(targets, preds, average='micro')
    f1_score_macro = metrics.f1_score(targets, preds, average='macro')
    auc_score_micro = metrics.roc_auc_score(targets, probabs, average='micro')
    auc_score_macro = metrics.roc_auc_score(targets, probabs, average='macro')
    precision_at_ks, p5_scores = precision_at_k(targets, probabs)

    logging.info(f"{dtset} Accuracy: {accuracy}")
    logging.info(f"{dtset} f1 score (micro): {f1_score_micro}")
    logging.info(f"{dtset} f1 score (macro): {f1_score_macro}")
    logging.info(f"{dtset} auc score (micro): {auc_score_micro}")
    logging.info(f"{dtset} auc score (macro): {auc_score_macro}")
    logging.info(f"{dtset} precision at ks [1, 5, 8, 10, 15]: {precision_at_ks}\n")

    print(f"\n{dtset} accuracy: {accuracy}"
          f"\n{dtset} f1 score (micro): {f1_score_micro}"
          f"\n{dtset} f1 score (macro): {f1_score_macro}"
          f"\n{dtset} auc score (micro): {auc_score_micro}"
          f"\n{dtset} auc score (macro): {auc_score_macro}"
          f"\n{dtset} precision at ks [1, 5, 8, 10, 15]: {precision_at_ks}")

    if dtset == 'test' and full_attn_weights:
        hype = '_'.join([f'{k}_{v}' for k, v in hyper_params._asdict().items()])
        save_predictions(probabs, targets, dtset, hype)
        # print(len(p5_scores), len(targets), len(full_hadm_ids), len(full_attn_weights))
        full_attn_weights = np.array(full_attn_weights)
        sorted_idx = np.argsort(p5_scores)[::-1]
        sorted_pred = np.argsort(probabs)[:, ::-1]
        top5_preds = sorted_pred[:, :5]
        with open(f'../results/{dtset}_attn_weights_{hype}.txt', 'w') as fout:
            for idx in sorted_idx:
                fout.write(f'idx: {idx}\n')
                fout.write(f'{full_hadm_ids[idx]};{p5_scores[idx]}\n')
                fout.write(f'{sorted_pred[idx]}\n')
                fout.write(f'{targets[idx, sorted_pred[idx]]}\n')
                fout.write(f'{preds[idx, sorted_pred[idx]]}\n')
                fout.write(f'{probabs[idx, sorted_pred[idx]]}\n')
                weights = full_attn_weights[idx, top5_preds[idx]]
                for wlist in weights:
                    line = ' '.join([str(val) for val in wlist])
                    fout.write(line+'\n')

