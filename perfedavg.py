import argparse
from mimetypes import init
import torch
from torch.utils.data import DataLoader, random_split
import os, random
from copy import deepcopy
from pickle_dataset import PickleDataset
import tqdm, logging

from utils import get_args, AverageMeter, get_optimizer, get_logger
from model import EmnistCNN
from fedavg1 import LocalTrainer

class LocalFineTuner:
    def __init__(self, optimizer_type, optimizer_args, criterion,
            epochs, batch_size, device) -> None:
        self.dataset = PickleDataset("femnist", pickle_root='../leaf/pickle_datasets')
        self.optimizer_type = optimizer_type
        self.optimizer_args = optimizer_args
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
    
    def train(self, model, client_id):
        trainloader = DataLoader(
            self.dataset.get_dataset_pickle("train", client_id), self.batch_size
        )

        initial_params = deepcopy(list(model.parameters()))
        optimizer = get_optimizer(model, self.optimizer_type, self.optimizer_args)
        self._train(model, trainloader, optimizer, self.epochs)

        gradients = []
        for old_param, new_param in zip(initial_params, model.parameters()):
            gradients.append(old_param.data - new_param.data)
        
        # weight = torch.tensor(len(trainloader.sampler))
        return gradients

    def _train(self, model, dataloader, optimizer, epochs):
        model.train()
        for _ in range(epochs):
            for x, y in dataloader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                outputs = model(x)
                loss = self.criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class PerLocalTrainer(LocalTrainer):
    def __init__(self, optimizer_type, optimizer_args, criterion,
            epochs, batch_size, pers_round, device) -> None:
        super().__init__(optimizer_type, optimizer_args, criterion,
            epochs, batch_size, pers_round, device)
        self.pers_round = pers_round

    def per_evaluate(self, model, test_clients):
        init_acc, per_acc = AverageMeter(), AverageMeter()
        for client in test_clients:
            acc1, acc2 = self._per_eval(deepcopy(model), client)
            print('per_eval:', acc1.average(), acc2.average())
            init_acc.update(acc1.sum, acc1.count)
            per_acc.update(acc2.sum, acc2.count)
        return init_acc.average(), per_acc.average()

    def _per_eval(self, model, client_id):
        dataset = self.dataset.get_dataset_pickle("test", client_id)
        trainset, valset = random_split(
            dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
        )
        trainloader = DataLoader(trainset, self.batch_size)
        valloader = DataLoader(valset, self.batch_size)
        init_loss, init_acc = self._eval(model, self.criterion, valloader)

        # TODO (adam)
        per_optimizer = get_optimizer(
            model, self.optimizer_type, self.optimizer_args
        )
        self._train(model, trainloader, per_optimizer, self.pers_round)
        per_loss, per_acc = self._eval(model, self.criterion, valloader)

        # return (init_loss, init_acc), (per_loss, per_acc)
        return init_acc, per_acc
    
    def _eval(self, model, criterion, valloader):
        model.eval()
        loss = AverageMeter()
        acc = AverageMeter()

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(device=self.device)
                labels = labels.to(device=self.device)

                outputs = model(inputs)
                l = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                loss.update(l.item(), len(labels))
                acc.update(torch.sum(predicted.eq(labels)).item(), len(labels))

        return loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    logger = get_logger(filename='./log.txt', enable_console=True)
    
    random.seed(0)
    torch.manual_seed(0)

    if args.cuda == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.cuda)

    global_model = EmnistCNN().to(device)
    global_optimizer = get_optimizer(
        global_model, "sgd", dict(lr=args.server_lr, momentum=0.9)
    )
    criterion = torch.nn.CrossEntropyLoss()

    # seperate clients
    num_training_clients = int(0.8 * args.client_num_in_total)
    training_clients_id_list = range(num_training_clients)
    num_testing_clients = args.client_num_in_total - num_training_clients
    test_clients_id_list = range(num_testing_clients)

    trainer = PerLocalTrainer(
        optimizer_type="sgd",
        optimizer_args=dict(lr=args.local_lr),
        criterion=criterion,
        epochs=args.inner_loops,
        batch_size=args.batch_size,
        pers_round=args.pers_round,
        device=device
    )
    
    test_clients = random.sample(test_clients_id_list, 40)

    # FedAvg training
    for e in range(args.epochs):
        selected_clients = random.sample(
            training_clients_id_list, args.client_num_per_round
        )
        all_client_weights = []
        all_client_gradients = []

        for client_id in tqdm.tqdm(selected_clients, desc="global round [{}]".format(e)):
            weight, grads = trainer.train(deepcopy(global_model), client_id)
            all_client_weights.append(weight)
            all_client_gradients.append(grads)

        # FedAvg aggregation (momentum SGD)
        global_optimizer.zero_grad()
        weights_sum = sum(all_client_weights)
        all_client_weights = [weight / weights_sum for weight in all_client_weights]
        for weight, grads in zip(all_client_weights, all_client_gradients):
            for param, grad in zip(global_model.parameters(), grads):
                if param.grad is None:
                    param.grad = torch.zeros(
                        param.size(), requires_grad=True, device=device
                    )
                param.grad.data.add_(grad.data * weight)
        global_optimizer.step()

        if e % 20 == 0:
            loss, acc = trainer.evaluate(deepcopy(global_model), test_clients)
            logging.info("epoch {0}: {1:.5f}, {2:.5f}".format(e, loss, acc))

    # eval
    test_clients_large = random.sample(test_clients_id_list, 400)
    init_acc, per_acc = trainer.per_evaluate(deepcopy(global_model), test_clients_large)
    logging.info("init_acc: {0:.5f}, per_acc: {1:.5f}".format(init_acc, per_acc))

    # Fine-tune
    logger.info("Fine-tune")
    server_optimizer = get_optimizer(
        global_model, "sgd", dict(lr=args.fine_tune_server_lr)
    )
    fine_tuner = LocalFineTuner(
        optimizer_type="adam",
        optimizer_args=dict(lr=args.fine_tune_local_lr, betas=(0, 0.999)),
        criterion=criterion,
        epochs=args.fine_tune_inner_loops,
        batch_size=args.batch_size,
        device=device,
    )

    for e in range(args.fine_tune_outer_loops):
        all_clients_gradients = []
        selected_clients = random.sample(
            training_clients_id_list, args.client_num_per_round
        )
        for client_id in selected_clients:
            # send model to clients and retrieve gradients
            grads = fine_tuner.train(deepcopy(global_model), client_id)
            all_clients_gradients.append(grads)

        # aggregate grads and update model
        server_optimizer.zero_grad()
        for grads in all_clients_gradients:
            for param, grad in zip(global_model.parameters(), grads):
                if param.grad is None:
                    param.grad = torch.zeros(
                        param.size(), requires_grad=True, device=device
                    )
                param.grad.data.add_(grad.data)
        for param in global_model.parameters():
            param.grad.data.div_(len(selected_clients))
        server_optimizer.step()
    logger.info("Fine-tune end")

    # final eval
    init_acc, per_acc = trainer.per_evaluate(global_model, test_clients_large)
    logging.info("init_acc: {0:.5f}, per_acc: {1:.5f}".format(init_acc, per_acc))
