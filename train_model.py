
from utils import *
import torch.nn as nn

CUDA = torch.cuda.is_available()


def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        out = net(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train


def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, data_val, label_val, subject, fold):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)

    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    model = get_model(args)
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()


    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0

    timer = Timer()
    patient = args.patient
    counter = 0

    for epoch in range(1, args.max_epoch + 1):

        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            save_model('candidate')
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))
    # save the training log file
    save_name = 'trlog' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = osp.join(args.save_path, experiment_setting, 'log_train')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))

    return trlog['max_acc'], trlog['F1']


def test(args, data, label, reproduce, subject, fold):
    set_up(args)
    seed_all(args.random_seed)
    test_loader = get_dataloader(data, label, args.batch_size)

    model = get_model(args)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    if reproduce:
        model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
        data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
        experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
        load_path_final = osp.join(args.save_path, experiment_setting, data_type, model_name_reproduce)
        model.load_state_dict(torch.load(load_path_final))
    else:
        model.load_state_dict(torch.load(args.load_path_final))
    loss, pred, act = predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn
    )
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act


def combine_train(args, data, label, subject, fold, target_acc):
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)
    seed_all(args.random_seed)
    train_loader = get_dataloader(data, label, args.batch_size)
    model = get_model(args)
    if CUDA:
        model = model.cuda()
    model.load_state_dict(torch.load(args.load_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate*1e-1)

    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch_cmb + 1):
        loss, pred, act = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer
        )
        acc, f1, _ = get_metrics(y_pred=pred, y_true=act)
        print('Stage 2 : epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss, acc, f1))

        if acc >= target_acc or epoch == args.max_epoch_cmb:
            print('early stopping!')
            save_model('final_model')
            # save model here for reproduce
            model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
            data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
            experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
            save_path = osp.join(args.save_path, experiment_setting, data_type)
            ensure_path(save_path)
            model_name_reproduce = osp.join(save_path, model_name_reproduce)
            torch.save(model.state_dict(), model_name_reproduce)
            break

        trlog['train_loss'].append(loss)
        trlog['train_acc'].append(acc)

        print('ETA:{}/{} SUB:{} TRIAL:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))

    save_name = 'trlog_comb' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = osp.join(args.save_path, experiment_setting, 'log_train_cmb')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))
