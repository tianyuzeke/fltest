import argparse
import torch
import logging
from collections import OrderedDict, defaultdict
import numpy as np
import torch.utils.data
from tqdm import trange
from pathlib import Path
import sys, random, os, json
from _utils import get_args
from models import CNNHyper, CNNTarget
from dataset import gen_random_loaders

sys.path.append('..')

from utils import get_logger

class LocalTrainer:
    def __init__(self, args, net, device):
        self.train_loaders, self.val_loaders, self.test_loaders = gen_random_loaders(
            args.data_name, args.data_path, args.num_nodes, args.batch_size, args.classes_per_node)

        self.device = device
        self.args = args
        self.net = net
        self.criteria = torch.nn.CrossEntropyLoss()

    def __len__(self):
        return self.n_nodes

    def train(self, weights, client_id):
        self.net.load_state_dict(weights)
        self.net.train()
        inner_state = OrderedDict({k: t.data for k, t in weights.items()})
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.args.inner_lr, momentum=.9, weight_decay=self.args.inner_wd)

        for i in range(self.args.inner_steps):
            
            batch = next(iter(self.train_loaders[client_id]))
            img, label = tuple(t.to(self.device) for t in batch)

            pred = self.net(img)
            loss = self.criteria(pred, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 50)

            optimizer.step()

        final_state = self.net.state_dict()

        return OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})
    
    @torch.no_grad()
    def evalute(self, weights, client_id, split):
        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            eval_data = trainer.test_loaders[client_id]
        elif split == 'val':
            eval_data = trainer.val_loaders[client_id]
        else:
            eval_data = trainer.train_loaders[client_id]
        
        self.net.load_state_dict(weights)

        for x, y in eval_data:
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred = self.net(x)
            running_loss += self.criteria(pred, y).item()
            # print(pred, running_loss)
            running_correct += pred.argmax(1).eq(y).sum().item()
            running_samples += len(y)
        return running_loss/(len(eval_data) + 1), running_correct, running_samples

def evaluate(hnet, trainer, clients, split):
    results = defaultdict(lambda: defaultdict(list))
    hnet.eval()

    for client_id in clients:     
        weights = hnet(torch.tensor([client_id], dtype=torch.long).to(device))
        running_loss, running_correct, running_samples = trainer.evalute(weights, client_id, split)

        results[client_id]['loss'] = running_loss
        results[client_id]['correct'] = running_correct
        results[client_id]['total'] = running_samples
    
    total_correct = sum([val['correct'] for val in results.values()])
    total_samples = sum([val['total'] for val in results.values()])
    avg_loss = np.mean([val['loss'] for val in results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in results.values()]

    return results, avg_loss, avg_acc, all_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    # logger = get_logger(filename='./log.txt', enable_console=True)
    
    random.seed(0)
    torch.manual_seed(0)

    if args.cuda == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.cuda)

    if args.data_name == 'cifar10':
        args.classes_per_node = 2
    else:
        args.classes_per_node = 10

    embed_dim = args.embed_dim
    n_embeddings = 20
    embed_dim = 600

    if embed_dim == -1:
        # logging.info("auto embedding size")
        embed_dim = int(1 + args.num_nodes / 4)

    if args.data_name == "cifar10":
        hnet = CNNHyper(args.num_nodes, n_embeddings, embed_dim, n_kernels=args.n_kernels)
        net = CNNTarget(n_kernels=args.n_kernels)
    elif args.data_name == "cifar100":
        hnet = CNNHyper(args.num_nodes, n_embeddings, embed_dim, n_kernels=args.n_kernels, out_dim=100)
        net = CNNTarget(n_kernels=args.n_kernels, out_dim=100)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")

    hnet = hnet.to(device)
    net = net.to(device)

    trainer = LocalTrainer(args, net, device)

    embed_lr = args.embed_lr if args.embed_lr is not None else args.lr
    optimizers = {
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=args.lr, momentum=0.9, weight_decay=args.wd
        ),
        'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr)
    }
    optimizer = optimizers[args.optim]
    criteria = torch.nn.CrossEntropyLoss()

    ################
    # init metrics #
    ################
    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1

    # embd = hnet.embeddings.weight.clone()
    # coeff = [e.clone() for e in hnet.coeff]

    results = defaultdict(list)
    for step in trange(args.num_steps):
        hnet.train()

        # delta_theta = OrderedDict({k: 0 for k in weights.keys()})
        delta_theta = w_sum = None

        for i in range(args.clients_per_round):

            # select client at random
            client_id = random.choice(range(args.num_nodes))

            # produce & load local network weights
            weights = hnet(torch.tensor([client_id], dtype=torch.long).to(device))
            net.load_state_dict(weights)

            delta = trainer.train(weights, client_id)

            final_state = net.state_dict()

            # calculating delta theta
            if delta_theta == None:
                delta_theta = delta
                w_sum = weights
            else:
                for k in delta.keys():
                    delta_theta[k] += delta[k]
                    w_sum[k] += weights[k]
        
        optimizer.zero_grad()
        # calculating phi gradient
        hnet_grads = torch.autograd.grad(
            list([w/args.clients_per_round for w in w_sum.values()]), hnet.parameters(), 
            grad_outputs=[v/args.clients_per_round for v in delta_theta.values()], allow_unused=True)

        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

        # A = ((embd - hnet.embeddings.weight)**2).view(-1).sum()
        # B = ((coeff[client_id] - hnet.coeff[client_id])**2).view(-1).sum()
        # print('***')
        # print("A: %s, B: %s" % (A, B))
        # print('***')

        if step % args.eval_every == 0:
            last_eval = step
            step_results, avg_loss, avg_acc, all_acc = evaluate(hnet, trainer, range(args.num_nodes), split="test")
            # logging.info(f"\nStep: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
            print(f"Step: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

            results['test_avg_loss'].append(avg_loss)
            results['test_avg_acc'].append(avg_acc)

            _, val_avg_loss, val_avg_acc, _ = evaluate(hnet, trainer, range(args.num_nodes), split="val")
            if best_acc < val_avg_acc:
                best_acc = val_avg_acc
                best_step = step
                test_best_based_on_step = avg_acc
                test_best_min_based_on_step = np.min(all_acc)
                test_best_max_based_on_step = np.max(all_acc)
                test_best_std_based_on_step = np.std(all_acc)

            results['val_avg_loss'].append(val_avg_loss)
            results['val_avg_acc'].append(val_avg_acc)
            results['best_step'].append(best_step)
            results['best_val_acc'].append(best_acc)
            results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
            results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
            results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
            results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

    if step != last_eval:
        _, val_avg_loss, val_avg_acc, _ = evaluate(hnet, trainer, range(args.num_nodes), split="val")
        step_results, avg_loss, avg_acc, all_acc = evaluate(hnet, trainer, range(args.num_nodes), split="test")
        # logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
        print(f"Step: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

        results['test_avg_loss'].append(avg_loss)
        results['test_avg_acc'].append(avg_acc)

        if best_acc < val_avg_acc:
            best_acc = val_avg_acc
            best_step = step
            test_best_based_on_step = avg_acc
            test_best_min_based_on_step = np.min(all_acc)
            test_best_max_based_on_step = np.max(all_acc)
            test_best_std_based_on_step = np.std(all_acc)

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)
        results['best_step'].append(best_step)
        results['best_val_acc'].append(best_acc)
        results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
        results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
        results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
        results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    if args.suffix:
        suffix = '_' + args.suffix
    with open(str(save_path / f"results_cr_{args.clients_per_round}_ins_{args.inner_steps}_es_{embed_dim}{args.suffix}.json"), "w") as file:
        json.dump(results, file, indent=4)


